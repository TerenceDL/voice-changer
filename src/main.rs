
use eframe::egui::{self, Frame, Margin, Vec2};
use std::process::{Child, Command, Stdio};

const ALSA_IN: &str = "hw:CARD=ATR2xUSB,DEV=0";
const ALSA_OUT: &str = "default";

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
                "-t","alsa", ALSA_IN, "-t","alsa", ALSA_OUT,
                "pitch","-400","bass","+10","reverb",
                "echo","0.8","0.9","1000","0.3",
            ],
            Preset::Alien => vec![
                "-t","alsa", ALSA_IN, "-t","alsa", ALSA_OUT,
                "pitch","+150","flanger",
                "chorus","0.7","0.9","55","0.4","0.25","2",
            ],
            Preset::Chipmunk => vec![
                "-t","alsa", ALSA_IN, "-t","alsa", ALSA_OUT,
                "pitch","+500","treble","+6",
            ],
            Preset::RadioHall => vec![
                "-t","alsa", ALSA_IN, "-t","alsa", ALSA_OUT,
                "highpass","300","lowpass","3400",
                "compand","0.3,1","6:-70,-60,-20 -5","-10","-90","0.2",
                "reverb","50",
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
        match Command::new("sox")
            .args(preset.sox_args())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                self.status = format!("Running: {}", preset.label());
                self.child = Some(child);
            }
            Err(e) => self.status = format!("Failed to start SoX: {}", e),
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
        apply_compact_style(ctx);

        // Slim top bar: status left, Quit right
        egui::TopBottomPanel::top("top_bar")
            .show_separator_line(false)
            .exact_height(86.0)
            .frame(Frame::default().inner_margin(Margin::symmetric(6,6)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.small(self.status.clone());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_sized(Vec2::new(90.0, 32.0), egui::Button::new("Quit")).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                });
            });

        // Central area: compute geometry, then center manually
        egui::CentralPanel::default()
            .frame(Frame::default().inner_margin(Margin::symmetric(8, 8)))
            .show(ctx, |ui| {
                let avail   = ui.available_size();          // size after top bar
                let gap     = ui.spacing().item_spacing.x;
                let title_h = 28.0;

                // Largest square button that fits both width and height:
                let side_max_x = (avail.x - gap) / 2.0;
                let side_max_y = (avail.y - title_h - 8.0 - gap) / 2.0;
                let side       = side_max_x.min(side_max_y).clamp(90.0, 160.0);
                let sz         = Vec2::splat(side);

                // Total block height (title + two rows) for vertical centering:
                let block_h = title_h + 8.0 + (2.0 * side + gap);
                let top_pad = ((avail.y - block_h) * 0.5).max(0.0);

                // *** Horizontal centering: compute exact left margin for a row ***
                let row_width = 2.0 * side + gap;
                let left_pad  = ((avail.x - row_width) * 0.5).max(0.0);

                // Move down to vertical center
                ui.add_space(top_pad);

                // Title centered
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.heading("Select a Mode");
                    ui.add_space(8.0);
                });

                // Row 1 — manual left pad + buttons
                ui.horizontal(|ui| {
                    ui.add_space(left_pad);
                    if ui.add_sized(sz, egui::Button::new("Deep Robot")).clicked() {
                        self.start_preset(Preset::DeepRobot);
                    }
                    ui.add_space(gap);
                    if ui.add_sized(sz, egui::Button::new("Alien")).clicked() {
                        self.start_preset(Preset::Alien);
                    }
                });

                ui.add_space(gap);

                // Row 2 — same manual centering
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

                // Keep block centered if there’s leftover vertical space
                let remaining = ui.available_size().y;
                if remaining > 0.0 { ui.add_space(remaining); }
            });
    }
}

fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions::default();
    // Fullscreen is simplest on the Pi; switch to windowed if you prefer.
    native_options.viewport = egui::ViewportBuilder::default().with_fullscreen(true);

    eframe::run_native(
        "Voice Changer",
        native_options,
        Box::new(|_cc| Ok(Box::new(VoiceChangerApp::new()))),
    )
}

