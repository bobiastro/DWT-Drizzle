#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod config;
mod license;
mod theme;

fn main() -> eframe::Result<()> {
    // 自证模式：dwt-drizzle.exe --dump-pages <目录>
    let args: Vec<String> = std::env::args().skip(1).collect();
    match app::parse_dump_pages(&args) {
        Ok(Some(dir)) => app::set_dump_dir(dir),
        Ok(None) => {}
        Err(e) => {
            eprintln!("参数错误：{e}");
            eprintln!("用法：dwt-drizzle.exe [--dump-pages <目录>]");
            std::process::exit(2);
        }
    }
    let title = format!("DWT · Drizzle {}", license::VERSION);
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(title.clone())
            .with_inner_size([1180.0, 940.0])
            .with_min_inner_size([1000.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(|cc| Ok(Box::new(app::DrizzleApp::new(cc)))),
    )
}
