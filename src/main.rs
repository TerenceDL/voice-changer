// // src/main.rs
// use eframe::egui;
// use std::process::{Child, Command, Stdio};

// const ALSA_IN: &str = "plughw:CARD=ATR2xUSB,DEV=0";
// const ALSA_OUT: &str = "default";

// #[derive(Clone, Copy, Debug)]
// enum Preset {
//     DeepRobot,
//     Alien,
//     Chipmunk,
//     RadioHall,
// }

// impl Preset {
//     fn label(&self) -> &'static str {
//         match self {
//             Preset::DeepRobot => "Deep Robot",
//             Preset::Alien => "Alien",
//             Preset::Chipmunk => "Chipmunk",
//             Preset::RadioHall => "Radio + Hall",
//         }
//     }

//     fn sox_args(&self) -> Vec<&'static str> {
//         match self {
//             Preset::DeepRobot => vec![
//                 "-t","alsa", ALSA_IN,
//                 "-t","alsa", ALSA_OUT,
//                 "pitch","-400",
//                 "bass","+10",
//                 "reverb",
//                 "echo","0.8","0.9","1000","0.3",
//             ],
//             Preset::Alien => vec![
//                 "-t","alsa", ALSA_IN,
//                 "-t","alsa", ALSA_OUT,
//                 "pitch","+150",
//                 "flanger",
//                 "chorus","0.7","0.9","55","0.4","0.25","2",
//             ],
//             Preset::Chipmunk => vec![
//                 "-t","alsa", ALSA_IN,
//                 "-t","alsa", ALSA_OUT,
//                 "pitch","+500",
//                 "treble","+6",
//             ],
//             Preset::RadioHall => vec![
//                 "-t","alsa", ALSA_IN,
//                 "-t","alsa", ALSA_OUT,
//                 "highpass","300",
//                 "lowpass","3400",
//                 "compand","0.3,1","6:-70,-60,-20 -5","-10","-90","0.2",
//                 "reverb","50",
//             ],
//         }
//     }
// }

// struct VoiceChangerApp {
//     child: Option<Child>,
//     status: String,
// }

// impl VoiceChangerApp {
//     fn new() -> Self { Self { child: None, status: "Idle".into() } }

//     fn start_preset(&mut self, preset: Preset) {
//         self.stop_current();
//         let args = preset.sox_args();
//         match Command::new("sox")
//             .args(args)
//             .stdin(Stdio::null())
//             .stdout(Stdio::null())
//             .stderr(Stdio::piped())
//             .spawn()
//         {
//             Ok(child) => {
//                 self.status = format!("Running preset: {}", preset.label());
//                 self.child = Some(child);
//             }
//             Err(e) => {
//                 self.status = format!("Failed to start SoX: {}", e);
//             }
//         }
//     }

//     fn stop_current(&mut self) {
//         if let Some(mut c) = self.child.take() {
//             let _ = c.kill();
//             let _ = c.wait();
//         }
//         self.status = "Stopped".into();
//     }
// }

// impl Drop for VoiceChangerApp {
//     fn drop(&mut self) {
//         if let Some(mut c) = self.child.take() {
//             let _ = c.kill();
//             let _ = c.wait();
//         }
//     }
// }

// impl eframe::App for VoiceChangerApp {
//     fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
//         egui::CentralPanel::default().show(ctx, |ui| {
//             ui.heading("Pi Voice Changer");
//             ui.label(&self.status);
//             ui.separator();

//             ui.horizontal(|ui| {
//                 if ui.button("Deep Robot").clicked() { self.start_preset(Preset::DeepRobot); }
//                 if ui.button("Alien").clicked() { self.start_preset(Preset::Alien); }
//                 if ui.button("Chipmunk").clicked() { self.start_preset(Preset::Chipmunk); }
//                 if ui.button("Radio + Hall").clicked() { self.start_preset(Preset::RadioHall); }
//             });

//             ui.separator();
//             if ui.button("Stop").clicked() { self.stop_current(); }
//             ui.add_space(10.0);
//             ui.label("Mic in:");
//             ui.monospace(ALSA_IN);
//             ui.label("Out:");
//             ui.monospace(ALSA_OUT);
//         });
//     }
// }

// fn main() -> eframe::Result<()> {
//     let native_options = eframe::NativeOptions::default();
//     eframe::run_native(
//         "Pi Voice Changer",
//         native_options,
//         Box::new(|_cc| Ok(Box::new(VoiceChangerApp::new()))),
//     )
// }

// src/main.rs
use eframe::egui;
use std::process::{Child, Command, Stdio};

const ALSA_IN: &str = "hw:CARD=ATR2xUSB,DEV=0";
const ALSA_OUT: &str = "default";

// Pre-recorded samples: (Label, Path). Put your files in an `assets/` folder next to the binary.
const SAMPLE_FILES: &[(&str, &str)] = &[
    ("Airhorn", "assets/airhorn.mp3"),
    ("Laser",   "assets/laser.wav"),
    ("Crowd",   "assets/crowd.wav"),
    ("Beep",    "assets/beep.wav"),
];

#[derive(Clone, Copy, Debug)]
enum Preset {
    DeepRobot,
    Alien,
    Chipmunk,
    RadioHall,
}

impl Preset {
    fn label(&self) -> &'static str {
        match self {
            Preset::DeepRobot => "Deep Robot",
            Preset::Alien => "Alien",
            Preset::Chipmunk => "Chipmunk",
            Preset::RadioHall => "Radio + Hall",
        }
    }

    fn sox_args(&self) -> Vec<&'static str> {
        match self {
            Preset::DeepRobot => vec![
                "-t","alsa", ALSA_IN,
                "-t","alsa", ALSA_OUT,
                "pitch","-400",
                "bass","+10",
                "reverb",
                "echo","0.8","0.9","1000","0.3",
            ],
            Preset::Alien => vec![
                "-t","alsa", ALSA_IN,
                "-t","alsa", ALSA_OUT,
                "pitch","+150",
                "flanger",
                "chorus","0.7","0.9","55","0.4","0.25","2",
            ],
            Preset::Chipmunk => vec![
                "-t","alsa", ALSA_IN,
                "-t","alsa", ALSA_OUT,
                "pitch","+500",
                "treble","+6",
            ],
            Preset::RadioHall => vec![
                "-t","alsa", ALSA_IN,
                "-t","alsa", ALSA_OUT,
                "highpass","300",
                "lowpass","3400",
                "compand","0.3,1","6:-70,-60,-20 -5","-10","-90","0.2",
                "reverb","50",
            ],
        }
    }
}

struct VoiceChangerApp {
    child: Option<Child>,
    status: String,
    buffer_bytes: i32,
    pulse_latency_ms: i32,
    pulse_sink: String,
}

impl VoiceChangerApp {
    fn new() -> Self {
        Self {
            child: None,
            status: "Idle".into(),
            buffer_bytes: 2048,
            pulse_latency_ms: 40,
            pulse_sink: "bluez_output.40_C1_F6_82_0E_97.1".to_string(),
        }
    }

    fn start_preset(&mut self, preset: Preset) {
        self.stop_current();
        let args = preset.sox_args();
        match Command::new("sox")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                self.status = format!("Running preset: {}", preset.label());
                self.child = Some(child);
            }
            Err(e) => {
                self.status = format!("Failed to start SoX: {}", e);
            }
        }
    }

    fn play_sample(&mut self, label: &str, path: &str) {
        self.stop_current();
        match Command::new("sox")
            .args([path, "-t", "alsa", ALSA_OUT])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                self.status = format!("Playing sample: {}", label);
                self.child = Some(child);
            }
            Err(e) => {
                self.status = format!("Failed to play {}: {}", label, e);
            }
        }
    }

    fn start_low_latency_bt(&mut self) {
        self.stop_current();
        let buf = self.buffer_bytes.to_string();
        let lat = self.pulse_latency_ms.to_string();
        let mut cmd = Command::new("sox");
        cmd.env("PULSE_LATENCY_MSEC", lat);
        if !self.pulse_sink.is_empty() {
            cmd.env("PULSE_SINK", self.pulse_sink.clone());
        }
        let args = vec![
            "-t","alsa", ALSA_IN,
            "-e","signed-integer","-b","16","-r","48000","-c","1",
            "-t","pulseaudio","default",
            "--buffer", &buf,
            "remix","1,1","pitch","-300","bass","+6",
        ];
        match cmd.args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn() {
            Ok(child) => {
                self.status = "Running: Low Latency (Bluetooth)".into();
                self.child = Some(child);
            }
            Err(e) => {
                self.status = format!("Failed to start SoX (BT): {}", e);
            }
        }
    }

    fn start_low_latency_wired(&mut self) {
        self.stop_current();
        let buf = self.buffer_bytes.to_string();
        let args = vec![
            "-t","alsa", ALSA_IN,
            "-e","signed-integer","-b","16","-r","48000","-c","1",
            "-t","alsa","hw:CARD=vc4hdmi0,DEV=0",
            "-e","signed-integer","-b","16","-r","48000","-c","1",
            "--buffer", &buf,
            "pitch","-300","bass","+6",
        ];
        match Command::new("sox").args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn() {
            Ok(child) => {
                self.status = "Running: Low Latency (Wired)".into();
                self.child = Some(child);
            }
            Err(e) => {
                self.status = format!("Failed to start SoX (Wired): {}", e);
            }
        }
    }

    fn stop_current(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.status = "Stopped".into();
    }
}

impl Drop for VoiceChangerApp {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl eframe::App for VoiceChangerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Pi Voice Changer");
            ui.label(&self.status);
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Deep Robot").clicked() { self.start_preset(Preset::DeepRobot); }
                if ui.button("Alien").clicked() { self.start_preset(Preset::Alien); }
                if ui.button("Chipmunk").clicked() { self.start_preset(Preset::Chipmunk); }
                if ui.button("Radio + Hall").clicked() { self.start_preset(Preset::RadioHall); }
            });

            ui.separator();
            ui.heading("Low Latency");
            ui.horizontal(|ui| {
                if ui.button("Bluetooth (Pulse)").clicked() { self.start_low_latency_bt(); }
                if ui.button("Wired (HDMI0)").clicked() { self.start_low_latency_wired(); }
            });
            ui.horizontal(|ui| {
                ui.label("Buffer bytes:");
                let mut b = self.buffer_bytes;
                if ui.add(egui::Slider::new(&mut b, 512..=8192).step_by(512.0)).changed() {
                    self.buffer_bytes = b;
                }
                ui.label("Pulse latency (ms):");
                let mut l = self.pulse_latency_ms;
                if ui.add(egui::Slider::new(&mut l, 20..=120).step_by(5.0)).changed() {
                    self.pulse_latency_ms = l;
                }
            });
            ui.horizontal(|ui| {
                ui.label("PULSE_SINK:");
                ui.text_edit_singleline(&mut self.pulse_sink);
            });

            ui.separator();
            ui.heading("Samples");
            ui.horizontal_wrapped(|ui| {
                for (label, path) in SAMPLE_FILES.iter().copied() {
                    if ui.button(label).clicked() {
                        self.play_sample(label, path);
                    }
                }
            });

            ui.separator();
            if ui.button("Stop").clicked() { self.stop_current(); }
            ui.add_space(10.0);
            ui.label("Mic in:");
            ui.monospace(ALSA_IN);
            ui.label("Out:");
            ui.monospace(ALSA_OUT);
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Pi Voice Changer",
        native_options,
        Box::new(|_| Ok(Box::new(VoiceChangerApp::new()))),
    )
}
