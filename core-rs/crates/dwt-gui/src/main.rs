#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod config;
mod license;
mod theme;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("DWT · Drizzle")
            .with_inner_size([1180.0, 940.0])
            .with_min_inner_size([1000.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "DWT · Drizzle",
        options,
        Box::new(|cc| Ok(Box::new(app::DrizzleApp::new(cc)))),
    )
}
