use std::sync::Arc;

use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, FontId, Frame, Label,
    Margin, RichText, Sense, Stroke, TextStyle, Ui, Vec2,
};

pub const BG: Color32 = Color32::from_rgb(0xf5, 0xf9, 0xfc);
pub const CARD: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
pub const BORDER: Color32 = Color32::from_rgb(0xd8, 0xe6, 0xf2);
pub const ACCENT: Color32 = Color32::from_rgb(0x0e, 0x6f, 0xb8);
pub const ACCENT_D: Color32 = Color32::from_rgb(0x0a, 0x5a, 0x96);
pub const FG: Color32 = Color32::from_rgb(0x0d, 0x22, 0x33);
pub const FG2: Color32 = Color32::from_rgb(0x5c, 0x74, 0x88);
pub const CYAN: Color32 = Color32::from_rgb(0x21, 0xa7, 0xc4);
pub const NAVY: Color32 = Color32::from_rgb(0x0b, 0x33, 0x50);
pub const LOG_BG: Color32 = Color32::from_rgb(0x0f, 0x15, 0x20);
pub const LOG_FG: Color32 = Color32::from_rgb(0xcb, 0xd5, 0xe1);
pub const OK: Color32 = Color32::from_rgb(0x16, 0x7a, 0x3c);
pub const ERR: Color32 = Color32::from_rgb(0xb9, 0x1c, 0x1c);

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    // Prefer a real Windows CJK face.  SimHei has a single face, so it is more
    // predictable than a TTC collection for ab_glyph.
    for (name, path) in [
        ("cjk", r"C:\Windows\Fonts\simhei.ttf"),
        ("cjk2", r"C:\Windows\Fonts\msyh.ttc"),
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(name.to_owned(), Arc::new(FontData::from_owned(bytes)));
            if let Some(f) = fonts.families.get_mut(&FontFamily::Proportional) {
                f.insert(0, name.into());
            }
            if let Some(f) = fonts.families.get_mut(&FontFamily::Monospace) {
                f.push(name.into());
            }
            break;
        }
    }
    if let Ok(bytes) = std::fs::read(r"C:\Windows\Fonts\consola.ttf") {
        fonts.font_data.insert("consola".into(), Arc::new(FontData::from_owned(bytes)));
        if let Some(f) = fonts.families.get_mut(&FontFamily::Monospace) {
            f.insert(0, "consola".into());
        }
    }
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    let mut v = egui::Visuals::light();
    v.panel_fill = BG;
    v.window_fill = CARD;
    v.faint_bg_color = BG;
    v.extreme_bg_color = Color32::from_rgb(0xee, 0xf5, 0xfa);
    v.override_text_color = Some(FG);
    v.selection.bg_fill = Color32::from_rgb(0xcf, 0xe8, 0xf8);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.noninteractive.bg_fill = CARD;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, FG);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.inactive.bg_fill = CARD;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, FG);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.hovered.bg_fill = Color32::from_rgb(0xe9, 0xf4, 0xfb);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.active.bg_fill = Color32::from_rgb(0xd9, 0xed, 0xf9);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT_D);
    v.window_stroke = Stroke::new(1.0_f32, BORDER);
    v.window_corner_radius = egui::CornerRadius::same(8);
    style.visuals = v;
    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.interact_size.y = 24.0;
    style.text_styles.insert(TextStyle::Body, FontId::new(14.0, FontFamily::Proportional));
    style.text_styles.insert(TextStyle::Button, FontId::new(14.0, FontFamily::Proportional));
    style.text_styles.insert(TextStyle::Small, FontId::new(11.0, FontFamily::Proportional));
    style.text_styles.insert(TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace));
    style.text_styles.insert(TextStyle::Heading, FontId::new(20.0, FontFamily::Proportional));
    ctx.set_style(style);
}

pub fn card(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui)) {
    Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(Margin::same(11))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).size(12.0).strong().color(FG2));
            ui.add_space(6.0);
            add(ui);
        });
    ui.add_space(8.0);
}

pub fn hint(ui: &mut Ui, text: &str) {
    ui.add(Label::new(RichText::new(text).size(11.0).color(FG2)).wrap());
}

/// Single-line editor with a visible input box on the near-white theme.
pub fn line_edit(value: &mut String) -> egui::TextEdit<'_> {
    egui::TextEdit::singleline(value)
        .frame(true)
        .margin(Vec2::new(6.0, 3.0))
        .background_color(Color32::from_rgb(0xf7, 0xfb, 0xff))
}

pub fn field_label(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).color(FG));
}

pub fn text_row(ui: &mut Ui, label: &str, value: &mut String, hint_text: &str) {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 24.0], Label::new(RichText::new(label).color(FG)));
        ui.add_sized([170.0, 24.0], line_edit(value));
    });
    hint_indented(ui, hint_text);
}

pub fn path_row(ui: &mut Ui, label: &str, value: &mut String, hint_text: &str, folder: bool) {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 24.0], Label::new(RichText::new(label).color(FG)));
        let width = (ui.available_width() - 88.0).max(160.0);
        ui.add_sized([width, 24.0], line_edit(value));
        if ui.button("浏览…").clicked() {
            let picked = if folder {
                rfd::FileDialog::new().pick_folder()
            } else {
                rfd::FileDialog::new().add_filter("XISF", &["xisf"]).pick_file()
            };
            if let Some(p) = picked {
                *value = p.display().to_string();
            }
        }
    });
    hint_indented(ui, hint_text);
}

pub fn hint_indented(ui: &mut Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.add_space(158.0);
        ui.add(Label::new(RichText::new(text).size(11.0).color(FG2)).wrap());
    });
}

pub fn choice_row(
    ui: &mut Ui,
    label: &str,
    value: &mut String,
    options: &[(&str, &str)],
    hint_text: &str,
) {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 24.0], Label::new(RichText::new(label).color(FG)));
        combo(ui, label, value, options);
    });
    hint_indented(ui, hint_text);
}

pub fn combo(ui: &mut Ui, id: &str, value: &mut String, options: &[(&str, &str)]) {
    let current = options
        .iter()
        .find(|(code, _)| *code == value.as_str())
        .map(|(_, label)| *label)
        .unwrap_or(value.as_str());
    egui::ComboBox::from_id_salt(id)
        .selected_text(current)
        .width(200.0)
        .show_ui(ui, |ui| {
            for (code, label) in options {
                if ui.selectable_label(value == code, *label).clicked() {
                    *value = (*code).to_string();
                }
            }
        });
}

pub fn tab_button(ui: &mut Ui, selected: bool, label: &str) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        FontId::proportional(15.0),
        if selected { ACCENT_D } else { FG2 },
    );
    let size = Vec2::new(galley.size().x + 30.0, 34.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let bg = if selected { CARD } else { Color32::TRANSPARENT };
    ui.painter().rect_filled(rect, egui::CornerRadius::same(8), bg);
    if selected {
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.left() + 15.0, rect.bottom() - 3.0),
                Vec2::new(rect.width() - 30.0, 2.0),
            ),
            egui::CornerRadius::same(1),
            ACCENT,
        );
    }
    let color = if selected { ACCENT_D } else { FG2 };
    ui.painter().galley(
        rect.center() - galley.size() * 0.5,
        galley,
        color,
    );
    resp
}

pub fn primary_button(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text).color(Color32::WHITE).strong())
        .fill(ACCENT)
        .stroke(Stroke::NONE)
}

pub fn danger_button(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text).color(ERR)).fill(CARD).stroke(Stroke::new(1.0_f32, ERR))
}

pub fn state_color(state: &str) -> Color32 {
    match state {
        "运行中" => CYAN,
        "完成" => OK,
        "失败" | "已停止" => ERR,
        _ => FG2,
    }
}

/// Open a URL without adding a browser dependency.
pub fn open_url(url: &str) {
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
}

pub fn check_row(ui: &mut Ui, label: &str, value: &mut bool, hint_text: &str) {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 24.0], Label::new(RichText::new(label).color(FG)));
        ui.checkbox(value, "");
    });
    hint_indented(ui, hint_text);
}

// ── 未实现参数 ────────────────────────────────────────────────────────────────
// 下面这些参数在 Python 版里有实现，Rust 引擎目前不消费它们的值。控件必须留在
// 界面上（面板布局要与 Python 版逐项对照），但置灰 + 标签标注 + 悬停说明，
// 免得用户以为「拨了没反应」是 bug。

/// 未实现控件的标签后缀。
pub const NOT_SUPPORTED: &str = "（Rust 版暂不支持）";

fn unsupported_scope(ui: &mut Ui, label: &str, tip: &str, row: impl FnOnce(&mut Ui, &str)) {
    let text = format!("{label}{NOT_SUPPORTED}");
    let hover = format!("{tip}\n\nRust 版暂不支持该参数：改动这里不会影响输出。");
    let inner = ui.scope(|ui| {
        ui.disable();
        row(ui, &text);
    });
    inner.response.on_hover_text(hover);
}

/// 置灰的 `text_row`（Rust 版未实现该参数）。
pub fn unsupported_text(ui: &mut Ui, label: &str, value: &mut String, hint_text: &str) {
    unsupported_scope(ui, label, hint_text, |ui, text| {
        text_row(ui, text, value, hint_text)
    });
}

/// 置灰的 `path_row`（Rust 版未实现该参数）。
pub fn unsupported_path(
    ui: &mut Ui,
    label: &str,
    value: &mut String,
    hint_text: &str,
    folder: bool,
) {
    unsupported_scope(ui, label, hint_text, |ui, text| {
        path_row(ui, text, value, hint_text, folder)
    });
}

/// 置灰的 `check_row`（Rust 版未实现该参数）。
pub fn unsupported_check(ui: &mut Ui, label: &str, value: &mut bool, hint_text: &str) {
    unsupported_scope(ui, label, hint_text, |ui, text| {
        check_row(ui, text, value, hint_text)
    });
}

/// 置灰的 `choice_row`（Rust 版未实现该参数）。
pub fn unsupported_choice(
    ui: &mut Ui,
    label: &str,
    value: &mut String,
    options: &[(&str, &str)],
    hint_text: &str,
) {
    unsupported_scope(ui, label, hint_text, |ui, text| {
        choice_row(ui, text, value, options, hint_text)
    });
}
