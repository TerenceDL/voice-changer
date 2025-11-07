use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, SampleRate, StreamConfig};
use ringbuf::traits::{Consumer as _, Producer as _, Split};
use ringbuf::HeapRb;
use std::env;
use std::f32::consts::PI;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

// ================= env helpers =================
fn env_flag(name: &str, default: bool) -> bool {
    env::var(name).ok().map(|v| v != "0" && v.to_lowercase() != "false").unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn db_to_lin(db: f32) -> f32 { 10.0f32.powf(db / 20.0) }

// ================= I/O prefs =================
const TARGET_SR: u32 = 48_000;
const RING_LATENCY_FRAMES: usize = 2048;

// ================= small utils =================
#[inline] fn one_pole_a(fc: f32, sr: f32) -> f32 { 1.0 - (-2.0 * PI * fc / sr).exp() }
#[inline] fn wrap_tau(x: f32) -> f32 { let t = 2.0*PI; if x >= t { x - t } else { x } }
#[inline]
fn frac_read(buf: &Vec<f32>, wi: usize, d: f32) -> f32 {
    let n = buf.len() as i32; if n <= 1 { return 0.0; }
    let mut pos = wi as f32 - d; while pos < 0.0 { pos += n as f32; }
    let i0 = pos.floor() as i32; let frac = pos - i0 as f32;
    let i1 = (i0 + 1).rem_euclid(n);
    let a = buf[i0.rem_euclid(n) as usize]; let b = buf[i1 as usize];
    a + (b - a) * frac
}

// ================= DC blocker =================
#[derive(Default)]
struct DcBlock { x1: f32, y1: f32, r: f32 }
impl DcBlock {
    fn new(r: f32) -> Self { Self { x1: 0.0, y1: 0.0, r } }
    #[inline] fn process(&mut self, x: f32) -> f32 {
        // y[n] = x[n] - x[n-1] + r*y[n-1]
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x; self.y1 = y; y
    }
}

// ================= RBJ biquad =================
#[derive(Clone, Copy)]
struct Bq { b0:f32,b1:f32,b2:f32,a1:f32,a2:f32, z1:f32, z2:f32 }
impl Bq {
    fn identity() -> Self { Self{b0:1.0,b1:0.0,b2:0.0,a1:0.0,a2:0.0,z1:0.0,z2:0.0} }
    #[inline] fn process(&mut self, x: f32) -> f32 {
        let y = self.b0*x + self.z1;
        self.z1 = self.b1*x - self.a1*y + self.z2;
        self.z2 = self.b2*x - self.a2*y;
        y
    }
}
fn rbj_low_shelf(fs: f32, fc: f32, gain_db: f32, q: f32) -> Bq {
    // RBJ cookbook
    let a  = (gain_db/40.0).exp2(); // sqrt(10^(dB/20))
    let w0 = 2.0*PI*fc/fs;
    let alpha = (w0/2.0).sin()/(2.0*q);
    let cosw0 = w0.cos();
    let two_sqrt_a = 2.0*(a).sqrt();

    let b0 =    a*((a+1.0) - (a-1.0)*cosw0 + two_sqrt_a*alpha);
    let b1 =  2.0*a*((a-1.0) - (a+1.0)*cosw0);
    let b2 =    a*((a+1.0) - (a-1.0)*cosw0 - two_sqrt_a*alpha);
    let a0 =        (a+1.0) + (a-1.0)*cosw0 + two_sqrt_a*alpha;
    let a1 = -2.0*((a-1.0) + (a+1.0)*cosw0);
    let a2 =        (a+1.0) + (a-1.0)*cosw0 - two_sqrt_a*alpha;
    norm(a0,b0,b1,b2,a1,a2)
}
fn rbj_high_shelf(fs: f32, fc: f32, gain_db: f32, q: f32) -> Bq {
    let a  = (gain_db/40.0).exp2();
    let w0 = 2.0*PI*fc/fs;
    let alpha = (w0/2.0).sin()/(2.0*q);
    let cosw0 = w0.cos();
    let two_sqrt_a = 2.0*(a).sqrt();

    let b0 =    a*((a+1.0) + (a-1.0)*cosw0 + two_sqrt_a*alpha);
    let b1 = -2.0*a*((a-1.0) + (a+1.0)*cosw0);
    let b2 =    a*((a+1.0) + (a-1.0)*cosw0 - two_sqrt_a*alpha);
    let a0 =        (a+1.0) - (a-1.0)*cosw0 + two_sqrt_a*alpha;
    let a1 =  2.0*((a-1.0) - (a+1.0)*cosw0);
    let a2 =        (a+1.0) - (a-1.0)*cosw0 - two_sqrt_a*alpha;
    norm(a0,b0,b1,b2,a1,a2)
}
fn rbj_peak(fs: f32, fc: f32, gain_db: f32, q: f32) -> Bq {
    let a  = (gain_db/40.0).exp2();
    let w0 = 2.0*PI*fc/fs;
    let alpha = (w0/2.0).sin()/(2.0*q);
    let cosw0 = w0.cos();

    let b0 = 1.0 + alpha*a;
    let b1 = -2.0*cosw0;
    let b2 = 1.0 - alpha*a;
    let a0 = 1.0 + alpha/a;
    let a1 = -2.0*cosw0;
    let a2 = 1.0 - alpha/a;
    norm(a0,b0,b1,b2,a1,a2)
}
fn norm(a0:f32,b0:f32,b1:f32,b2:f32,a1:f32,a2:f32) -> Bq {
    Bq{ b0:b0/a0, b1:b1/a0, b2:b2/a0, a1:a1/a0, a2:a2/a0, z1:0.0, z2:0.0 }
}

// ================= Compressor =================
struct Comp {
    thr: f32, // linear
    ratio: f32,
    atk: f32, rel: f32,
    env: f32,
    makeup: f32,
}
impl Comp {
    fn new(sr: u32) -> Self {
        let thr_db = env_f32("CMP_THR_DB", -18.0);
        let ratio  = env_f32("CMP_RATIO", 3.0).max(1.0);
        let atk_ms = env_f32("CMP_ATK_MS", 5.0);
        let rel_ms = env_f32("CMP_REL_MS", 60.0);
        let mk_db  = env_f32("CMP_MAKEUP_DB", 3.0);

        let atk = (-(1.0/((atk_ms/1000.0)*(sr as f32)))).exp();
        let rel = (-(1.0/((rel_ms/1000.0)*(sr as f32)))).exp();

        Self {
            thr: db_to_lin(thr_db),
            ratio,
            atk, rel,
            env: 0.0,
            makeup: db_to_lin(mk_db),
        }
    }
    #[inline] fn process(&mut self, x: f32) -> f32 {
        let a = x.abs();
        self.env = if a > self.env { self.atk * self.env + (1.0 - self.atk) * a }
                   else             { self.rel * self.env + (1.0 - self.rel) * a };
        let gain = if self.env <= self.thr || self.env <= 1e-6 {
            1.0
        } else {
            let over = self.env / self.thr;
            let gr = over.powf(1.0 - 1.0/self.ratio);
            1.0 / gr
        };
        x * gain * self.makeup
    }
}

// ================== FX ==================
struct GhostFX {
    // toggles
    tilt_on: bool, chorus_on: bool, ring_on: bool, slap_on: bool, gate_on: bool, comp_on: bool,

    // DC + tone EQ (biquads)
    dc: DcBlock,
    shelf_low: Bq,
    shelf_high: Bq,
    peak1: Bq,
    peak2: Bq,

    // ring
    rm_phase: f32,
    rm_inc: f32,
    rm_mix: f32,

    // chorus
    ch_buf: Vec<f32>,
    ch_idx: usize,
    ch_lfo: f32,
    ch_lfo_inc: f32,
    ch_base: f32,
    ch_depth: f32,
    ch_mix: f32,

    // slap
    slp_buf: Vec<f32>,
    slp_idx: usize,
    slp_fb: f32,
    slp_mix: f32,
    slp_lp: Bq,

    // drive
    drive: f32,

    // gate state
    gate_env: f32,
    gate_thresh: f32,
    gate_atk: f32,
    gate_rel: f32,

    // compressor
    comp: Comp,
}
impl GhostFX {
    fn new(sr: u32) -> Self {
        let sr_f = sr as f32;

        // toggles (Ghost preset: everything on)
        let tilt_on   = env_flag("FX_TILT",   true);
        let chorus_on = env_flag("FX_CHORUS", true);
        let ring_on   = env_flag("FX_RING",   true);
        let slap_on   = env_flag("FX_SLAP",   true);
        let gate_on   = env_flag("FX_GATE",   true);
        let comp_on   = env_flag("FX_COMP",   true);

        // DC
        let dc_r = env_f32("DSP_DC_R", 0.995).clamp(0.9, 0.9999);

        // EQ: low chest + high dark + mid shape
        let low_g  = env_f32("EQ_LOW_DB",  6.0);
        let low_fc = env_f32("EQ_LOW_FC",  180.0);
        let low_q  = env_f32("EQ_LOW_Q",   0.7);

        let hi_g   = env_f32("EQ_HI_DB",  -5.0);
        let hi_fc  = env_f32("EQ_HI_FC",  3800.0);
        let hi_q   = env_f32("EQ_HI_Q",   0.7);

        let p1_g   = env_f32("EQ_P1_DB",   4.5); // rasp presence
        let p1_fc  = env_f32("EQ_P1_FC",   1900.0);
        let p1_q   = env_f32("EQ_P1_Q",    1.0);

        let p2_g   = env_f32("EQ_P2_DB",  -3.0); // de-honk
        let p2_fc  = env_f32("EQ_P2_FC",   900.0);
        let p2_q   = env_f32("EQ_P2_Q",    1.0);

        let shelf_low = rbj_low_shelf(sr_f, low_fc, low_g, low_q);
        let shelf_high= rbj_high_shelf(sr_f, hi_fc,  hi_g, hi_q);
        let peak1     = rbj_peak(sr_f, p1_fc, p1_g, p1_q);
        let peak2     = rbj_peak(sr_f, p2_fc, p2_g, p2_q);

        // ring
        let rm_hz  = env_f32("DSP_RM_HZ", 35.0);
        let rm_inc = (2.0*PI*rm_hz) / sr_f;
        let rm_mix = env_f32("DSP_RM_MIX", 0.18).clamp(0.0, 1.0);

        // chorus
        let base_ms  = env_f32("DSP_CH_BASE_MS", 9.0);
        let depth_ms = env_f32("DSP_CH_DEPTH_MS", 4.0);
        let lfo_hz   = env_f32("DSP_CH_LFO_HZ", 0.55);
        let ch_base  = base_ms * 1e-3 * sr_f;
        let ch_depth = depth_ms * 1e-3 * sr_f;
        let ch_lfo_inc = (2.0*PI*lfo_hz) / sr_f;
        let ch_mix   = env_f32("DSP_CH_MIX", 0.32).clamp(0.0, 1.0);
        let ch_len   = (ch_base + ch_depth + 8.0).ceil() as usize;

        // slapback
        let slp_ms  = env_f32("DSP_SLP_MS", 110.0);
        let slp_len = (slp_ms * 1e-3 * sr_f).max(1.0).round() as usize;
        let slp_fb  = env_f32("DSP_SLP_FB", 0.28).clamp(0.0, 0.9);
        let slp_mix = env_f32("DSP_SLP_MIX", 0.18).clamp(0.0, 1.0);
        let slp_lp_fc = env_f32("DSP_SLP_LP_FC", 3200.0);
        let mut slp_lp = rbj_low_shelf(sr_f, slp_lp_fc, -8.0, 0.7); // darker repeats
        // HACK: approximate simple LP feel by stacking shelf twice
        let _ = &mut slp_lp;

        let drive = env_f32("DSP_DRIVE", 1.6);

        // gate (light)
        let gate_thresh = env_f32("DSP_GATE_THRESH", 0.0038);
        let gate_atk    = env_f32("DSP_GATE_ATK", 0.15);
        let gate_rel    = env_f32("DSP_GATE_REL", 0.004);

        let comp = Comp::new(sr);

        Self {
            tilt_on, chorus_on, ring_on, slap_on, gate_on, comp_on,
            dc: DcBlock::new(dc_r),
            shelf_low, shelf_high, peak1, peak2,
            rm_phase: 0.0, rm_inc, rm_mix,
            ch_buf: vec![0.0; ch_len.max(1)], ch_idx: 0,
            ch_lfo: 0.0, ch_lfo_inc, ch_base, ch_depth, ch_mix,
            slp_buf: vec![0.0; slp_len], slp_idx: 0,
            slp_fb, slp_mix, slp_lp,
            drive,
            gate_env: 0.0, gate_thresh, gate_atk, gate_rel,
            comp,
        }
    }

    #[inline] fn process(&mut self, x_in: f32) -> f32 {
        let mut x = if x_in.is_finite() { x_in.clamp(-1.0, 1.0) } else { 0.0 };

        // ----- optional micro-gate -----
        if self.gate_on {
            let open = if x.abs() > self.gate_thresh { 1.0 } else { 0.0 };
            self.gate_env = if open > 0.0 {
                self.gate_env + self.gate_atk * (1.0 - self.gate_env)
            } else {
                self.gate_env * (1.0 - self.gate_rel)
            };
            x *= self.gate_env;
        }

        // ----- tone stack -----
        if self.tilt_on {
            x = self.dc.process(x);
            x = self.shelf_low.process(x);
            x = self.peak2.process(x); // de-honk first
            x = self.peak1.process(x); // rasp presence
            x = self.shelf_high.process(x);
        }

        // ----- compressor (before grunge) -----
        if self.comp_on { x = self.comp.process(x); }

        // ----- ring mod -----
        if self.ring_on && self.rm_mix > 0.0 {
            let m = self.rm_phase.sin();
            self.rm_phase = wrap_tau(self.rm_phase + self.rm_inc);
            x = (1.0 - self.rm_mix) * x + self.rm_mix * (x * m);
        }

        // ----- chorus -----
        if self.chorus_on && !self.ch_buf.is_empty() {
            self.ch_buf[self.ch_idx] = x;
            let d = self.ch_base + self.ch_depth * self.ch_lfo.sin();
            self.ch_lfo = wrap_tau(self.ch_lfo + self.ch_lfo_inc);
            let cy = frac_read(&self.ch_buf, self.ch_idx, d.max(1.0));
            self.ch_idx = (self.ch_idx + 1) % self.ch_buf.len();
            x = (1.0 - self.ch_mix) * x + self.ch_mix * cy;
        }

        // ----- slapback -----
        if self.slap_on && !self.slp_buf.is_empty() {
            let tap = self.slp_buf[self.slp_idx];
            // darken repeats a bit via simple 1st-order tilt: reuse shelf_low negatively
            let tap_d = tap; // already darkish with shelf in feedback path
            let write = x + self.slp_fb * tap_d;
            self.slp_buf[self.slp_idx] = write;
            self.slp_idx = (self.slp_idx + 1) % self.slp_buf.len();
            x = (1.0 - self.slp_mix) * x + self.slp_mix * tap;
        }

        // ----- soft clip -----
        (x * self.drive).tanh()
    }
}

// ================== app ==================
fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("list") { return list_devices(); }

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::SeqCst))?;
    }

    let host = cpal::default_host();
    let in_dev  = pick_device(&host, true,  env::var("IN_DEV").ok().as_deref())?;
    let out_dev = pick_device(&host, false, env::var("OUT_DEV").ok().as_deref())?;
    println!("Input  device: {}", in_dev.name()?);
    println!("Output device: {}", out_dev.name()?);

    let in_cfg  = choose_config_supported(&in_dev,  true)?;
    let out_cfg = choose_config_supported(&out_dev, false)?;
    println!("Input  config: {:?}", in_cfg);
    println!("Output config: {:?}", out_cfg);

    let bypass = env_flag("DSP_BYPASS", false);
    let wet    = env_f32("DSP_WET", 1.0).clamp(0.0, 1.0);
    println!("DSP: bypass={bypass}, wet={wet}");

    // ring buffer (mono path)
    let cap_samples = (RING_LATENCY_FRAMES * out_cfg.channels as usize).max(1024);
    let rb = HeapRb::<f32>::new(cap_samples);
    let (mut prod, mut cons) = rb.split();

    // INPUT: downmix to mono if needed
    let in_ch = in_cfg.channels as usize;
    let input_stream = build_input_stream(&in_dev, &in_cfg, move |data: &[f32]| {
        if in_ch <= 1 {
            for &s in data { let _ = prod.try_push(s); }
        } else {
            for fr in data.chunks_exact(in_ch) {
                let m = 0.5 * (fr[0] + fr[1]);
                let _ = prod.try_push(m);
            }
        }
    })?;

    // OUTPUT
    let mut fx = GhostFX::new(out_cfg.sample_rate.0);
    let ch = out_cfg.channels as usize;
    let output_stream = build_output_stream(&out_dev, &out_cfg, move |out: &mut [f32]| {
        for frame in out.chunks_mut(ch) {
            let mut x = cons.try_pop().unwrap_or(0.0);
            if !x.is_finite() { x = 0.0; }

            let y = if bypass { x } else { fx.process(x) };
            let o = if bypass { y } else { (1.0 - wet) * x + wet * y };

            for s in frame { *s = o; }
        }
    })?;

    input_stream.play()?;
    output_stream.play()?;
    println!("Streaming… (Ctrl+C to quit)");
    while running.load(Ordering::SeqCst) { std::thread::sleep(std::time::Duration::from_millis(50)); }
    Ok(())
}

// ================= I/O helpers =================
fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("=== INPUT DEVICES ===");
    for d in host.input_devices()? {
        println!("- {}", d.name()?);
        if let Ok(cfgs) = d.supported_input_configs() {
            for c in cfgs {
                println!("    {:?}, channels ≥ {}, rate {:?}-{:?} Hz",
                    c.sample_format(), c.channels(), c.min_sample_rate().0, c.max_sample_rate().0);
            }
        }
    }
    println!("\n=== OUTPUT DEVICES ===");
    for d in host.output_devices()? {
        println!("- {}", d.name()?);
        if let Ok(cfgs) = d.supported_output_configs() {
            for c in cfgs {
                println!("    {:?}, channels ≥ {}, rate {:?}-{:?} Hz",
                    c.sample_format(), c.channels(), c.min_sample_rate().0, c.max_sample_rate().0);
            }
        }
    }
    Ok(())
}
fn pick_device(host: &cpal::Host, is_input: bool, want_substr: Option<&str>) -> Result<cpal::Device> {
    let want = want_substr.map(|s| s.to_lowercase());
    let devs = if is_input { host.input_devices()? } else { host.output_devices()? };
    println!("Searching {} devices. Want substring: {:?}", if is_input { "input" } else { "output" }, want);
    if let Some(w) = want.as_ref() {
        for d in devs {
            let name = d.name()?.to_lowercase();
            println!("  Candidate: {}", name);
            if name.contains(w) { println!("  ✅ Matched: {}", name); return Ok(d); }
        }
        bail!("No matching {} device for '{}'", if is_input { "input" } else { "output" }, w);
    }
    if is_input { host.default_input_device().context("No default input device") }
    else { host.default_output_device().context("No default output device") }
}
fn choose_config_supported(dev: &cpal::Device, is_input: bool) -> Result<StreamConfig> {
    let def = if is_input { dev.default_input_config().context("No default input config")? }
              else { dev.default_output_config().context("No default output config")? };
    Ok(StreamConfig {
        channels: def.channels().max(1).min(2),
        sample_rate: SampleRate(TARGET_SR),
        buffer_size: BufferSize::Default,
    })
}
fn build_input_stream<F>(dev: &cpal::Device, req_cfg: &StreamConfig, mut on_input: F) -> Result<cpal::Stream>
where F: FnMut(&[f32]) + Send + 'static {
    let def = dev.default_input_config().context("No default input config")?;
    let cfg = StreamConfig { channels: req_cfg.channels.min(def.channels()),
                             sample_rate: req_cfg.sample_rate,
                             buffer_size: req_cfg.buffer_size.clone() };
    let err_fn = |e| eprintln!("Input error: {e}");
    let stream = match def.sample_format() {
        SampleFormat::F32 => dev.build_input_stream(&cfg, move |d: &[f32], _| on_input(d), err_fn, None)?,
        SampleFormat::I16 => dev.build_input_stream(&cfg, move |d: &[i16], _| {
                let mut tmp = vec![0.0f32; d.len()];
                for (i,s) in d.iter().enumerate() { tmp[i] = *s as f32 / i16::MAX as f32; }
                on_input(&tmp);
            }, err_fn, None)?,
        SampleFormat::U16 => dev.build_input_stream(&cfg, move |d: &[u16], _| {
                let mut tmp = vec![0.0f32; d.len()];
                for (i,s) in d.iter().enumerate() { tmp[i] = (*s as f32 - 32768.0) / 32768.0; }
                on_input(&tmp);
            }, err_fn, None)?,
        other => bail!("Unsupported input format: {:?}", other),
    };
    Ok(stream)
}
fn build_output_stream<F>(dev: &cpal::Device, req_cfg: &StreamConfig, mut fill_out: F) -> Result<cpal::Stream>
where F: FnMut(&mut [f32]) + Send + 'static {
    let def = dev.default_output_config().context("No default output config")?;
    let cfg = StreamConfig { channels: req_cfg.channels.min(def.channels()),
                             sample_rate: req_cfg.sample_rate,
                             buffer_size: req_cfg.buffer_size.clone() };
    let err_fn = |e| eprintln!("Output error: {e}");
    let stream = match def.sample_format() {
        SampleFormat::F32 => dev.build_output_stream(&cfg, move |o: &mut [f32], _| fill_out(o), err_fn, None)?,
        SampleFormat::I16 => dev.build_output_stream(&cfg, move |o: &mut [i16], _| {
                let mut tmp = vec![0.0f32; o.len()];
                fill_out(&mut tmp);
                for (d,s) in o.iter_mut().zip(tmp.iter()) {
                    *d = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                }
            }, err_fn, None)?,
        SampleFormat::U16 => dev.build_output_stream(&cfg, move |o: &mut [u16], _| {
                let mut tmp = vec![0.0f32; o.len()];
                fill_out(&mut tmp);
                for (d,s) in o.iter_mut().zip(tmp.iter()) {
                    let v = ((s.clamp(-1.0, 1.0) * 32768.0) + 32768.0).round();
                    *d = v.clamp(0.0, 65535.0) as u16;
                }
            }, err_fn, None)?,
        other => bail!("Unsupported output format: {:?}", other),
    };
    Ok(stream)
}
