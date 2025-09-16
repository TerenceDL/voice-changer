use eframe::egui::{self, Frame, Margin, Vec2};
use std::io::Read;
use std::process::{Child, Command, Stdio};

const ALSA_IN: &str = "hw:CARD=ATR2xUSB,DEV=0";
const ALSA_OUT: &str = "default";

// small buffer keeps latency modest but avoids constant xruns
const SOX_BUFFER_BYTES: &str = "2048";

#[derive(Clone, Copy, Debug)]
enum Preset {
    DeepRobot,
    GhostFace,
    Chipmunk,
    RadioHall,
}

impl Preset {
    fn label(&self) -> &'static str {
        match self {
            Preset::DeepRobot => "Deep Robot",
            Preset::GhostFace => "Ghost Face",
            Preset::Chipmunk => "Chipmunk",
            Preset::RadioHall => "Radio + Hall",
        }
    }

    fn sox_args(&self) -> Vec<&'static str> {
        match self {
            // Low-latency deep voice: no echo/reverb
            Preset::DeepRobot => vec![
                "--buffer", SOX_BUFFER_BYTES,
                "-t", "alsa", ALSA_IN,
                "-t", "alsa", ALSA_OUT,
                "pitch", "-400",
                "bass",  "+10",
            ],

            // Ghost Face: slight pitch down + band-limit + compression + light grit
            Preset::GhostFace => vec![
                "--buffer", SOX_BUFFER_BYTES,
                "-t", "alsa", ALSA_IN,
                "-t", "alsa", ALSA_OUT,
                // keep it scary but intelligible
                "pitch", "-200",
                // telephone/radio band
                "highpass", "250",
                "lowpass",  "3400",
                // gentle broadcast-style compression (attack,decay | curve | out gain | init vol | delay)
                "compand", "0.3,1", "6:-70,-60,-20,-5", "-8", "-90", "0.2",
                // touch of grit (gain, color) — keep subtle
                "overdrive", "10", "20",
            ],

            // Bright, fast, no reverb
            Preset::Chipmunk => vec![
                "--buffer", SOX_BUFFER_BYTES,
                "-t", "alsa", ALSA_IN,
                "-t", "alsa", ALSA_OUT,
                "pitch",  "+500",
                "treble", "+6",
            ],

            // Radio band + compand + small hall
            Preset::RadioHall => vec![
                "--buffer", SOX_BUFFER_BYTES,
                "-t", "alsa", ALSA_IN,
                "-t", "alsa", ALSA_OUT,
                "highpass", "300",
                "lowpass",  "3400",
                // attack,decay | transfer points | gain | initial vol | delay
                "compand", "0.3,1", "6:-70,-60,-20,-5", "-10", "-90", "0.2",
                "reverb", "50",
            ],
        }
    }
}

struct VoiceChangerApp {
    child: Option<Child>,
    status: String,
}

impl VoiceChangerApp {
    fn new() -> Self {
        Self { child: None, status: "Ready".into() }
    }

    fn start_preset(&mut self, preset: Preset) {
        self.stop_current();

        // Build & spawn SoX
        match Command::new("sox")
            .args(preset.sox_args())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped()) // capture errors for UI
            .spawn()
        {
            Ok(child) => {
                self.status = format!("Running: {}", preset.label());
                self.child = Some(child);
            }
            Err(e) => {
                self.status = format!("Failed to start SoX: {}", e);
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

    /// Called each frame to see if SoX died early (e.g., parse error / device busy).
    fn poll_child_and_update_status(&mut self) {
        if let Some(child) = &mut self.child {
            if let Ok(Some(status)) = child.try_wait() {
                let mut err = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_string(&mut err);
                }
                if status.success() {
                    self.status = "Finished.".into();
                } else {
                    let last = err
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("SoX exited with error");
                    self.status = format!("SoX error: {}", last);
                }
                self.child = None;
            }
        }
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

fn apply_compact_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (egui::TextStyle::Heading,   egui::FontId::proportional(20.0)),
        (egui::TextStyle::Body,      egui::FontId::proportional(18.0)),
        (egui::TextStyle::Button,    egui::FontId::proportional(18.0)),
        (egui::TextStyle::Monospace, egui::FontId::monospace(14.0)),
        (egui::TextStyle::Small,     egui::FontId::proportional(12.0)),
    ].into();
    style.spacing.item_spacing   = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(10.0, 8.0);
    style.spacing.interact_size  = Vec2::splat(52.0);
    ctx.set_style(style);

    // Keep layout predictable on the Pi; remove if you prefer auto DPI.
    ctx.set_pixels_per_point(1.0);
}

impl eframe::App for VoiceChangerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // keep existing look/placement
        apply_compact_style(ctx);

        // check if the running preset died & surface any SoX error
        self.poll_child_and_update_status();

        // Top bar (unchanged placement)
        egui::TopBottomPanel::top("top_bar")
            .show_separator_line(false)
            .exact_height(86.0)
            .frame(Frame::default().inner_margin(Margin::symmetric(6, 6)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.small(self.status.clone());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_sized(Vec2::new(90.0, 32.0), egui::Button::new("Quit"))
                            .clicked()
                        {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                });
            });

        // Central area (unchanged placement): manually centered 2×2 grid under title
        egui::CentralPanel::default()
            .frame(Frame::default().inner_margin(Margin::symmetric(8, 8)))
            .show(ctx, |ui| {
                let avail   = ui.available_size();
                let gap     = ui.spacing().item_spacing.x;
                let title_h = 28.0;

                let side_max_x = (avail.x - gap) / 2.0;
                let side_max_y = (avail.y - title_h - 8.0 - gap) / 2.0;
                let side       = side_max_x.min(side_max_y).clamp(90.0, 160.0);
                let sz         = Vec2::splat(side);

                let block_h = title_h + 8.0 + (2.0 * side + gap);
                let top_pad = ((avail.y - block_h) * 0.5).max(0.0);

                let row_width = 2.0 * side + gap;
                let left_pad  = ((avail.x - row_width) * 0.5).max(0.0);

                ui.add_space(top_pad);

                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.heading("Select a Mode");
                    ui.add_space(8.0);
                });

                ui.horizontal(|ui| {
                    ui.add_space(left_pad);
                    if ui.add_sized(sz, egui::Button::new("Deep Robot")).clicked() {
                        self.start_preset(Preset::DeepRobot);
                    }
                    ui.add_space(gap);
                    if ui.add_sized(sz, egui::Button::new("Ghost Face")).clicked() {
                        self.start_preset(Preset::GhostFace);
                    }
                });

                ui.add_space(gap);

                ui.horizontal(|ui| {
                    ui.add_space(left_pad);
                    if ui.add_sized(sz, egui::Button::new("Chipmunk")).clicked() {
                        self.start_preset(Preset::Chipmunk);
                    }
                    ui.add_space(gap);
                    if ui.add_sized(sz, egui::Button::new("Radio+Hall")).clicked() {
                        self.start_preset(Preset::RadioHall);
                    }
                });

                let remaining = ui.available_size().y;
                if remaining > 0.0 { ui.add_space(remaining); }
            });
    }
}

fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = egui::ViewportBuilder::default().with_fullscreen(true);

    eframe::run_native(
        "Voice Changer",
        native_options,
        Box::new(|_cc| Ok(Box::new(VoiceChangerApp::new()))),
    )
}


