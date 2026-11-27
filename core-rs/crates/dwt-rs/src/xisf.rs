//! Minimal XISF 1.0 reader/writer: single-image, single-channel, planar Float32
//! (the subset the PWS stack needs). Byte-compatible with `xisf_io.py` for the
//! uncompressed layout produced by PixInsight/WBPP and by this tool.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::image::Img;

/// One `<FITSKeyword name="..." value="..." comment="..."/>` child.
#[derive(Debug, Clone, Default)]
pub struct FitsKeyword {
    pub name: String,
    /// Raw value as written: string keywords keep their surrounding quotes.
    pub value: String,
    pub comment: String,
}

impl FitsKeyword {
    pub fn new(name: &str, value: impl Into<String>, comment: &str) -> Self {
        Self { name: name.to_string(), value: value.into(), comment: comment.to_string() }
    }
    /// Value with the FITS string quotes stripped (what `_aggregate_fits_keywords`
    /// compares on: `str(v).strip("'\"")`).
    pub fn bare(&self) -> String {
        self.value.trim_matches(|c| c == '\'' || c == '"').to_string()
    }
}

/// One `<Property id="..." type="..." .../>` element, attributes kept verbatim so
/// the cross-frame aggregation can re-emit base64 vectors and structures as-is.
#[derive(Debug, Clone, Default)]
pub struct XisfProperty {
    pub id: String,
    pub ty: String,
    pub attributes: Vec<(String, String)>,
    pub text: Option<String>,
}

impl XisfProperty {
    pub fn value_prop(id: &str, ty: &str, value: &str) -> Self {
        Self {
            id: id.to_string(),
            ty: ty.to_string(),
            attributes: vec![
                ("id".into(), id.to_string()),
                ("type".into(), ty.to_string()),
                ("value".into(), value.to_string()),
            ],
            text: None,
        }
    }
    pub fn text_prop(id: &str, ty: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            ty: ty.to_string(),
            attributes: vec![("id".into(), id.to_string()), ("type".into(), ty.to_string())],
            text: Some(text.to_string()),
        }
    }
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    /// The `value` attribute (numeric/typed properties).
    pub fn value(&self) -> Option<&str> {
        self.attr("value")
    }
    /// Whatever carries the payload: the `value` attribute or the text content.
    pub fn payload(&self) -> Option<String> {
        self.value().map(|v| v.to_string()).or_else(|| self.text.clone())
    }
}

#[derive(Debug, Clone)]
pub struct XisfMeta {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub sample_format: String,
    pub color_space: String,
    pub pixel_storage: String,
    pub bounds: Option<(f32, f32)>,
    /// FITSKeyword children of `<Image>` in file order.
    pub keywords: Vec<FitsKeyword>,
    /// XISF `<Property>` children of `<Image>` (the CFA pattern lives here on
    /// PixInsight files that do not carry a FITS `BAYERPAT` keyword).
    pub properties: Vec<XisfProperty>,
}

impl XisfMeta {
    pub fn keyword(&self, name: &str) -> Option<&str> {
        self.keywords
            .iter()
            .find(|k| k.name.eq_ignore_ascii_case(name))
            .map(|k| k.value.as_str())
    }
    /// FITS string keyword without the surrounding quotes.
    pub fn keyword_str(&self, name: &str) -> Option<String> {
        self.keyword(name).map(|v| v.trim().trim_matches('\'').to_string())
    }
    pub fn keyword_f64(&self, name: &str) -> Option<f64> {
        self.keyword(name).and_then(|v| v.trim().trim_matches('\'').parse().ok())
    }
}

fn attr<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
    // attributes are `name="value"`; search with a leading space to avoid matching
    // a suffix of another attribute name
    let pat = format!(" {name}=\"");
    let start = xml.find(&pat)? + pat.len();
    let rest = &xml[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn xml_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let ent = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ]
        .into_iter()
        .find(|(e, _)| rest.starts_with(e));
        match ent {
            Some((e, c)) => {
                out.push(c);
                rest = &rest[e.len()..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// XML attribute escaping — ElementTree (`xisf_io`) escapes all five entities in
/// attribute values, so a byte-level header diff against the Python product needs
/// the same treatment.
fn xml_escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

fn xml_escape_text(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn read_xisf(path: &Path) -> Result<(XisfMeta, Img), String> {
    let (meta, flat, channels) = read_xisf_flat(path)?;
    if channels != 1 {
        return Err(format!(
            "{}: only single-channel images supported here (got {channels})",
            path.display()
        ));
    }
    let (w, h) = (meta.width, meta.height);
    Ok((meta, Img::from_vec(w, h, flat)))
}

/// Read a (possibly multi-channel) XISF and split it into one plane per channel.
/// Handles both `pixelStorage="Normal"` (interleaved HWC, what PixInsight writes
/// for RGB) and `"Planar"` (channel-major, used by this tool for mono).
pub fn read_xisf_planes(path: &Path) -> Result<(XisfMeta, Vec<Img>), String> {
    let (meta, flat, channels) = read_xisf_flat(path)?;
    let n = meta.width * meta.height;
    let mut planes = Vec::with_capacity(channels);
    if meta.pixel_storage.eq_ignore_ascii_case("Planar") {
        for ch in 0..channels {
            planes.push(Img::from_vec(meta.width, meta.height, flat[ch * n..(ch + 1) * n].to_vec()));
        }
    } else {
        for ch in 0..channels {
            let mut d = Vec::with_capacity(n);
            for p in 0..n {
                d.push(flat[p * channels + ch]);
            }
            planes.push(Img::from_vec(meta.width, meta.height, d));
        }
    }
    Ok((meta, planes))
}

/// Read an XISF and collapse any channel count to a single luminance plane
/// (mean of the finite channel values).  Used for reference canvases and for
/// `--match-scale`, where only the sky level / star field matters.
pub fn read_xisf_luma(path: &Path) -> Result<(XisfMeta, Img), String> {
    let (meta, flat, channels) = read_xisf_flat(path)?;
    let n = meta.width * meta.height;
    let mut d = vec![0.0f32; n];
    for p in 0..n {
        let base = p * channels;
        let mut acc = 0.0f64;
        let mut cnt = 0u32;
        for ch in 0..channels {
            let v = if meta.pixel_storage.eq_ignore_ascii_case("Planar") {
                flat[ch * n + p]
            } else {
                flat[base + ch]
            };
            if v.is_finite() {
                acc += v as f64;
                cnt += 1;
            }
        }
        d[p] = if cnt > 0 { (acc / cnt as f64) as f32 } else { f32::NAN };
    }
    let (w, h) = (meta.width, meta.height);
    Ok((meta, Img::from_vec(w, h, d)))
}

fn read_xisf_flat(path: &Path) -> Result<(XisfMeta, Vec<f32>, usize), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut head = [0u8; 16];
    f.read_exact(&mut head).map_err(|e| format!("read header: {e}"))?;
    if &head[..8] != b"XISF0100" {
        return Err(format!("{}: not an XISF 1.0 file", path.display()));
    }
    let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut xml_buf = vec![0u8; header_len];
    f.read_exact(&mut xml_buf).map_err(|e| format!("read xml: {e}"))?;
    let xml = String::from_utf8_lossy(&xml_buf).to_string();

    let image_start = xml.find("<Image").ok_or("no <Image> element")?;
    let image_end = image_start + xml[image_start..].find('>').ok_or("bad <Image>")?;
    let image_tag = &xml[image_start..image_end];

    let geometry = attr(image_tag, "geometry").ok_or("no geometry")?;
    let dims: Vec<usize> = geometry.split(':').filter_map(|s| s.parse().ok()).collect();
    if dims.len() < 2 {
        return Err(format!("bad geometry {geometry}"));
    }
    let (width, height) = (dims[0], dims[1]);
    let channels = if dims.len() > 2 { dims[2] } else { 1 };
    let sample_format = attr(image_tag, "sampleFormat").unwrap_or("Float32").to_string();
    let color_space = attr(image_tag, "colorSpace").unwrap_or("Gray").to_string();
    let pixel_storage = attr(image_tag, "pixelStorage").unwrap_or("Planar").to_string();
    let bounds = attr(image_tag, "bounds").and_then(|b| {
        let mut it = b.split(':');
        let lo = it.next()?.parse::<f32>().ok()?;
        let hi = it.next()?.parse::<f32>().ok()?;
        Some((lo, hi))
    });
    let location = attr(image_tag, "location").unwrap_or("");
    let compression = attr(image_tag, "compression");
    let byte_order = attr(image_tag, "byteOrder").unwrap_or("little");

    // Metadata/properties + FITSKeyword children live inside the `<Image>` element;
    // the `<Metadata>` block also carries `XISF:` properties, which are *not* the
    // per-frame instrument/signature set `_aggregate_properties` works on.
    let body_end = xml[image_start..]
        .find("</Image>")
        .map(|p| image_start + p)
        .unwrap_or(xml.len());
    let body = &xml[image_end..body_end];

    let mut keywords = Vec::new();
    let mut cursor = 0usize;
    while let Some(p) = body[cursor..].find("<FITSKeyword") {
        let s = cursor + p;
        let e = s + body[s..].find('>').unwrap_or(0);
        let tag = &body[s..e];
        if let Some(n) = attr(tag, "name") {
            keywords.push(FitsKeyword {
                name: n.to_string(),
                value: attr(tag, "value").map(xml_unescape).unwrap_or_default(),
                comment: attr(tag, "comment").map(xml_unescape).unwrap_or_default(),
            });
        }
        cursor = e.max(s + 1);
    }

    // XISF <Property> elements (PCL:CFASourcePattern etc.).  A property payload is
    // either the `value` attribute or the element text (String/TimePoint).
    let mut properties = Vec::new();
    let mut cursor = 0usize;
    while let Some(p) = body[cursor..].find("<Property") {
        let s = cursor + p;
        let close = body[s..].find('>').map(|q| s + q).unwrap_or(body.len());
        let tag = &body[s..close];
        if let Some(id) = attr(tag, "id") {
            let mut attributes: Vec<(String, String)> = Vec::new();
            let mut rest = tag;
            while let Some(eq) = rest.find("=\"") {
                let before = &rest[..eq];
                let name_start = before
                    .rfind(|c: char| c.is_whitespace())
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let name = &before[name_start..];
                let vstart = eq + 2;
                let vend = rest[vstart..].find('"').map(|q| vstart + q).unwrap_or(rest.len());
                if !name.is_empty() {
                    attributes.push((name.to_string(), xml_unescape(&rest[vstart..vend])));
                }
                rest = &rest[(vend + 1).min(rest.len())..];
            }
            let self_closing = tag.trim_end().ends_with('/');
            let text = if self_closing {
                None
            } else {
                let tstart = (close + 1).min(body.len());
                body[tstart..].find("</Property>").map(|q| xml_unescape(body[tstart..tstart + q].trim()))
            };
            let ty = attributes
                .iter()
                .find(|(k, _)| k == "type")
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            properties.push(XisfProperty { id: id.to_string(), ty, attributes, text });
        }
        cursor = close.max(s + 1);
    }

    let bytes_per_sample = match sample_format.as_str() {
        "Float32" => 4usize,
        "Float64" => 8,
        "UInt16" => 2,
        "UInt32" => 4,
        other => return Err(format!("unsupported sampleFormat {other}")),
    };
    let n = width * height;
    let nsamples = n * channels;
    let mut raw = vec![0u8; nsamples * bytes_per_sample];
    if let Some(rest) = location.strip_prefix("attachment:") {
        let parts: Vec<usize> = rest.split(':').filter_map(|s| s.parse().ok()).collect();
        let (pos, size) = (parts[0], *parts.get(1).unwrap_or(&0));
        f.seek(SeekFrom::Start(pos as u64)).map_err(|e| format!("seek: {e}"))?;
        if let Some(comp) = compression {
            // compression="zlib:encoded:item_size"
            let mut payload = vec![0u8; size];
            f.read_exact(&mut payload).map_err(|e| format!("read block: {e}"))?;
            if comp.starts_with("zlib") {
                let mut out = Vec::with_capacity(nsamples * bytes_per_sample);
                let mut dec = flate2::read::ZlibDecoder::new(&payload[..]);
                dec.read_to_end(&mut out).map_err(|e| format!("zlib: {e}"))?;
                if out.len() < raw.len() {
                    return Err(format!("zlib block too short: {} < {}", out.len(), raw.len()));
                }
                let want = raw.len();
                raw.copy_from_slice(&out[..want]);
            } else {
                return Err(format!("unsupported compression {comp}"));
            }
        } else {
            if size != raw.len() {
                return Err(format!("block size {size} != expected {}", raw.len()));
            }
            f.read_exact(&mut raw).map_err(|e| format!("read block: {e}"))?;
        }
    } else if location.starts_with("inline:") {
        let parts: Vec<usize> = location.strip_prefix("inline:").unwrap().split(':').filter_map(|s| s.parse().ok()).collect();
        let (pos, _size) = (parts[0], parts[1]);
        f.seek(SeekFrom::Start(pos as u64)).map_err(|e| format!("seek: {e}"))?;
        f.read_exact(&mut raw).map_err(|e| format!("read block: {e}"))?;
    } else {
        return Err(format!("unsupported location '{location}'"));
    }

    let mut data = Vec::with_capacity(nsamples);
    let big = byte_order == "big";
    match sample_format.as_str() {
        "Float32" => {
            for i in 0..nsamples {
                let b = [raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]];
                data.push(if big { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) });
            }
        }
        "Float64" => {
            for i in 0..nsamples {
                let mut b = [0u8; 8];
                b.copy_from_slice(&raw[8 * i..8 * i + 8]);
                let v = if big { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) };
                data.push(v as f32);
            }
        }
        "UInt16" => {
            for i in 0..nsamples {
                let b = [raw[2 * i], raw[2 * i + 1]];
                data.push(if big { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) } as f32);
            }
        }
        "UInt32" => {
            for i in 0..nsamples {
                let b = [raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]];
                data.push(if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) } as f32);
            }
        }
        _ => unreachable!(),
    }
    Ok((
        XisfMeta {
            width,
            height,
            channels,
            sample_format,
            color_space,
            pixel_storage,
            bounds,
            keywords,
            properties,
        },
        data,
        channels,
    ))
}

/// Values written into the output header. Mirrors `save_linear_xisf` in `core/pws.py`:
/// the three product keywords plus the cross-frame aggregated FITS keywords and
/// XISF properties (`_aggregate_fits_keywords` / `_aggregate_properties` / `_pws_props`).
pub struct OutMeta {
    pub total_exp: f64,
    pub ncombine: u32,
    pub stack_mode: String,
    pub keywords: Vec<FitsKeyword>,
    pub properties: Vec<XisfProperty>,
}

impl OutMeta {
    /// The three keywords the product always writes, in `save_linear_xisf` order.
    pub fn product_keywords(&self) -> [FitsKeyword; 3] {
        [
            FitsKeyword::new(
                "TOTALEXP",
                format!("{:.1}", self.total_exp),
                "total exposure seconds of combined frames",
            ),
            FitsKeyword::new("NCOMBINE", self.ncombine.to_string(), "frames combined"),
            // Python drops the mode text straight into the keyword list, so the
            // value reaches the header unquoted; keep the bytes identical.
            FitsKeyword::new(
                "STACKMODE",
                self.stack_mode.clone(),
                "partition-weighted stacking (linear output, no stretch)",
            ),
        ]
    }
}

fn render_keyword(out: &mut String, kw: &FitsKeyword) {
    out.push_str(&format!(
        "      <FITSKeyword name=\"{}\" value=\"{}\"",
        xml_escape_attr(&kw.name),
        xml_escape_attr(&kw.value)
    ));
    if !kw.comment.is_empty() {
        out.push_str(&format!(" comment=\"{}\"", xml_escape_attr(&kw.comment)));
    }
    out.push_str("/>\n");
}

fn render_property(out: &mut String, p: &XisfProperty) {
    let mut attrs = String::new();
    for (k, v) in &p.attributes {
        attrs.push_str(&format!(" {}=\"{}\"", k, xml_escape_attr(v)));
    }
    match &p.text {
        Some(t) if !t.is_empty() => out.push_str(&format!(
            "      <Property{attrs}>{}</Property>\n",
            xml_escape_text(t)
        )),
        _ => out.push_str(&format!("      <Property{attrs}/>\n")),
    }
}

pub fn write_xisf(path: &Path, img: &Img, meta: &OutMeta) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in &img.d {
        if v.is_finite() {
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
    }
    if !(hi > lo) {
        lo = 0.0;
        hi = 1.0;
    }
    let mut kws = String::new();
    for kw in meta.product_keywords() {
        render_keyword(&mut kws, &kw);
    }
    // Aggregated per-frame keywords (Python: `fits_keywords = [...] + aggregated`).
    for kw in &meta.keywords {
        if matches!(kw.name.to_ascii_uppercase().as_str(), "TOTALEXP" | "NCOMBINE" | "STACKMODE") {
            continue;
        }
        render_keyword(&mut kws, kw);
    }
    for p in &meta.properties {
        render_property(&mut kws, p);
    }

    // two-pass: the header length depends on the digit count of the data offset,
    // and the offset depends on the header length -> iterate until stable.
    let mut xml = String::new();
    let mut data_offset = 0usize;
    for _ in 0..8 {
        xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\n\
   <Image geometry=\"{w}:{h}:1\" sampleFormat=\"Float32\" bounds=\"{lo}:{hi}\" colorSpace=\"Gray\" \
location=\"attachment:{data_offset}:{nbytes}\" pixelStorage=\"Planar\">\n{kws}\
   </Image>\n\
</xisf>\n",
            w = img.w,
            h = img.h,
            nbytes = img.len() * 4,
        );
        let xml_end = 16 + xml.len();
        let pad = 4096 - (xml_end % 4096);
        let pad = if pad == 4096 { 0 } else { pad };
        let new_offset = xml_end + pad;
        if new_offset == data_offset {
            break;
        }
        data_offset = new_offset;
    }

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"XISF0100").map_err(|e| e.to_string())?;
    f.write_all(&(xml.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&[0u8, 0, 0, 0]).map_err(|e| e.to_string())?;
    f.write_all(xml.as_bytes()).map_err(|e| e.to_string())?;
    let xml_end = 16 + xml.len();
    let pad = 4096 - (xml_end % 4096);
    let pad = if pad == 4096 { 0 } else { pad };
    f.write_all(&vec![0u8; pad]).map_err(|e| e.to_string())?;
    f.write_all(bytemuck::cast_slice(&img.d)).map_err(|e| e.to_string())?;
    Ok(())
}

/// Write a 3-plane RGB image as an interleaved HWC `Float32` XISF, matching
/// what `xisf_io.py` produces for `PWS_<tag>_RGB.xisf`: no FITSKeyword block,
/// real data bounds, `pixelStorage="Normal"`, `colorSpace="RGB"`.
pub fn write_xisf_rgb(path: &Path, planes: &[Img]) -> Result<(), String> {
    if planes.len() != 3 {
        return Err(format!("write_xisf_rgb needs 3 planes, got {}", planes.len()));
    }
    let (w, h) = (planes[0].w, planes[0].h);
    if planes.iter().any(|p| p.w != w || p.h != h) {
        return Err("write_xisf_rgb: plane dimensions differ".into());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in planes {
        for v in &p.d {
            if v.is_finite() {
                if *v < lo {
                    lo = *v;
                }
                if *v > hi {
                    hi = *v;
                }
            }
        }
    }
    if !(hi > lo) {
        lo = 0.0;
        hi = 1.0;
    }
    let mut xml = String::new();
    let mut data_offset = 0usize;
    let nbytes = w * h * 3 * 4;
    for _ in 0..8 {
        xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\n\
   <Image geometry=\"{w}:{h}:3\" sampleFormat=\"Float32\" bounds=\"{lo}:{hi}\" colorSpace=\"RGB\" \
location=\"attachment:{data_offset}:{nbytes}\" pixelStorage=\"Normal\">\n\
   </Image>\n\
</xisf>\n"
        );
        let xml_end = 16 + xml.len();
        let pad = 4096 - (xml_end % 4096);
        let pad = if pad == 4096 { 0 } else { pad };
        let new_offset = xml_end + pad;
        if new_offset == data_offset {
            break;
        }
        data_offset = new_offset;
    }

    let mut interleaved = Vec::with_capacity(w * h * 3);
    for p in 0..w * h {
        interleaved.push(planes[0].d[p]);
        interleaved.push(planes[1].d[p]);
        interleaved.push(planes[2].d[p]);
    }

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"XISF0100").map_err(|e| e.to_string())?;
    f.write_all(&(xml.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&[0u8, 0, 0, 0]).map_err(|e| e.to_string())?;
    f.write_all(xml.as_bytes()).map_err(|e| e.to_string())?;
    let xml_end = 16 + xml.len();
    let pad = 4096 - (xml_end % 4096);
    let pad = if pad == 4096 { 0 } else { pad };
    f.write_all(&vec![0u8; pad]).map_err(|e| e.to_string())?;
    f.write_all(bytemuck::cast_slice(&interleaved)).map_err(|e| e.to_string())?;
    Ok(())
}

/// Write a `numpy.save` compatible `.npy` (v1.0, C-order, little-endian f32).
/// Used purely for cross-checking intermediate fields against the Python run.
pub fn write_npy_f32(path: &Path, data: &[f32], h: usize, w: usize) -> Result<(), String> {
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}");
    let mut header = b"\x93NUMPY\x01\x00".to_vec();
    // 8-byte magic + 2-byte header length + dict + spaces + '\n' must be a multiple of 64
    let mut body = dict.into_bytes();
    let needed = 10 + body.len() + 1;
    let rem = needed % 64;
    let pad = if rem == 0 { 0 } else { 64 - rem };
    body.extend(std::iter::repeat_n(b' ', pad));
    body.push(b'\n');
    debug_assert_eq!((10 + body.len()) % 64, 0);
    header.extend((body.len() as u16).to_le_bytes());
    header.extend(body);
    let mut f = File::create(path).map_err(|e| format!("create npy: {e}"))?;
    f.write_all(&header).map_err(|e| e.to_string())?;
    f.write_all(bytemuck::cast_slice(data)).map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Cross-frame metadata aggregation — port of `_aggregate_fits_keywords`,
// `_aggregate_properties` and the `PWS:` property set in `core/pws.py`.
// ---------------------------------------------------------------------------

/// Product-owned keywords: never passed through from the input frames.
const META_SKIP_KW: [&str; 3] = ["TOTALEXP", "NCOMBINE", "STACKMODE"];
/// Per-frame measurements (`PSF*`, `NOISE*`) only hold for a single frame.
const META_SKIP_KW_PREFIX: [&str; 2] = ["PSF", "NOISE"];

/// Python `float` repr, close enough for XISF `value` attributes: integral floats
/// keep the `.0` (`str(1200.0) == '1200.0'`), everything else is shortest round-trip.
pub fn py_num(v: f64) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "nan".into() } else if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// C's `%.6g` (Python's `fmt='%.6g'` in the weights table): six significant
/// digits, trailing zeros dropped, exponent form outside `1e-4 .. 1e6`.
pub fn py_g6(v: f64) -> String {
    if !v.is_finite() {
        return py_num(v);
    }
    if v == 0.0 {
        // C prints the sign of a negative zero; keep it so the table matches.
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let sign = if v < 0.0 { "-" } else { "" };
    let a = v.abs();
    let mut exp = a.log10().floor() as i32;
    // `%g` rounds to six significant digits *before* choosing the form, so
    // 999999.5 prints as `1e+06` and 9.9999999e-5 as `0.0001`.
    let mut mant = format!("{:.5}", a / 10f64.powi(exp));
    if mant.parse::<f64>().unwrap_or(0.0) >= 10.0 {
        exp += 1;
        mant = format!("{:.5}", a / 10f64.powi(exp));
    }
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    if exp < -4 || exp >= 6 {
        let m = mant.trim_end_matches('0').trim_end_matches('.').to_string();
        return format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    // Lay the six significant digits around the decimal point.
    let point = 1 + exp;
    let mut body = if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        digits
    } else {
        format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
    };
    if body.contains('.') {
        while body.ends_with('0') {
            body.pop();
        }
        if body.ends_with('.') {
            body.pop();
        }
    }
    format!("{sign}{body}")
}

pub fn median_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Minimal civil-time value with Python `datetime.isoformat()` output, so the
/// DATE-OBS/DATE-END strings are byte-identical to the Python product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IsoTime {
    /// Civil seconds since 1970-01-01T00:00:00 (the frame timestamp is UTC-naive).
    pub secs: i64,
    pub micros: u32,
    /// UTC offset in minutes, kept only when the input carried a timezone.
    pub offset_min: Option<i32>,
}

impl IsoTime {
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let sep = s.find(['T', ' '])?;
        let (date, mut time) = s.split_at(sep);
        time = &time[1..];
        let mut dit = date.split('-');
        let y: i64 = dit.next()?.parse().ok()?;
        let mo: i64 = dit.next()?.parse().ok()?;
        let d: i64 = dit.next()?.parse().ok()?;

        let mut offset_min = None;
        let mut tz_split = time.len();
        for (i, c) in time.char_indices() {
            if c == '+' || (c == '-' && i > 0) || c == 'Z' || c == 'z' {
                tz_split = i;
                break;
            }
        }
        let tz = &time[tz_split..];
        time = &time[..tz_split];
        if !tz.is_empty() {
            if tz.eq_ignore_ascii_case("z") {
                offset_min = Some(0);
            } else {
                let sign = if tz.starts_with('-') { -1 } else { 1 };
                let body: String = tz.chars().filter(|c| c.is_ascii_digit()).collect();
                let hh: i32 = body.get(0..2).and_then(|v| v.parse().ok()).unwrap_or(0);
                let mm: i32 = body.get(2..4).and_then(|v| v.parse().ok()).unwrap_or(0);
                offset_min = Some(sign * (hh * 60 + mm));
            }
        }

        let mut tit = time.split(':');
        let h: i64 = tit.next()?.parse().ok()?;
        let mi: i64 = tit.next().unwrap_or("0").parse().ok()?;
        let sec_part = tit.next().unwrap_or("0");
        let (whole, frac) = match sec_part.find('.') {
            Some(p) => (&sec_part[..p], &sec_part[p + 1..]),
            None => (sec_part, ""),
        };
        let sec: i64 = whole.parse().ok()?;
        let micros: u32 = if frac.is_empty() {
            0
        } else {
            let mut digits: String = frac.chars().take(6).collect();
            while digits.len() < 6 {
                digits.push('0');
            }
            digits.parse().ok()?
        };
        let days = days_from_civil(y, mo, d);
        Some(Self { secs: days * 86_400 + h * 3_600 + mi * 60 + sec, micros, offset_min })
    }

    pub fn add_seconds(&self, secs: f64) -> Self {
        let dm = (secs * 1e6).round() as i64;
        let total = self.secs * 1_000_000 + self.micros as i64 + dm;
        let (s, us) = (total.div_euclid(1_000_000), total.rem_euclid(1_000_000) as u32);
        Self { secs: s, micros: us, offset_min: self.offset_min }
    }

    /// `datetime.isoformat()`: `YYYY-MM-DDTHH:MM:SS[.ffffff][+HH:MM]`.
    pub fn to_iso(self) -> String {
        let days = self.secs.div_euclid(86_400);
        let rem = self.secs.rem_euclid(86_400);
        let (y, mo, d) = civil_from_days(days);
        let (h, mi, s) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);
        let mut out = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}");
        if self.micros > 0 {
            out.push_str(&format!(".{:06}", self.micros));
        }
        if let Some(off) = self.offset_min {
            let sign = if off < 0 { '-' } else { '+' };
            let a = off.abs();
            out.push_str(&format!("{sign}{:02}:{:02}", a / 60, a % 60));
        }
        out
    }
}

/// Howard Hinnant's `days_from_civil` (proleptic Gregorian, no libm/libc).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn push_grouped<'a>(
    order: &mut Vec<String>,
    groups: &mut Vec<Vec<&'a FitsKeyword>>,
    index: &mut std::collections::HashMap<String, usize>,
    kw: &'a FitsKeyword,
) {
    let name = kw.name.to_ascii_uppercase();
    let i = match index.get(&name) {
        Some(&i) => i,
        None => {
            order.push(name.clone());
            groups.push(Vec::new());
            let i = groups.len() - 1;
            index.insert(name, i);
            i
        }
    };
    groups[i].push(kw);
}

/// `_aggregate_fits_keywords`: a keyword present in *every* frame with an equal
/// value is passed through verbatim (value + comment of the first frame);
/// DATE-OBS becomes the earliest start and DATE-END the last start plus the
/// median exposure.  Keywords whose values differ are dropped — Python does not
/// invent a `VARIOUS` placeholder.
pub fn aggregate_fits_keywords(metas: &[XisfMeta], exptimes: &[f64]) -> Vec<FitsKeyword> {
    let n = metas.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<&FitsKeyword>> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for m in metas {
        for kw in &m.keywords {
            push_grouped(&mut order, &mut groups, &mut index, kw);
        }
    }
    let med_exp = median_f64(exptimes);
    let mut out = Vec::new();
    for (i, name) in order.iter().enumerate() {
        let g = &groups[i];
        if g.len() != n
            || META_SKIP_KW.contains(&name.as_str())
            || META_SKIP_KW_PREFIX.iter().any(|p| name.starts_with(p))
        {
            continue;
        }
        if name == "DATE-OBS" {
            let mut ts: Vec<IsoTime> = match g.iter().map(|k| IsoTime::parse(&k.bare())).collect::<Option<Vec<_>>>() {
                Some(v) => v,
                None => continue,
            };
            ts.sort();
            let first = ts[0];
            let last = *ts.last().unwrap();
            out.push(FitsKeyword::new(
                "DATE-OBS",
                format!("'{}'", first.to_iso()),
                "earliest frame start (UTC)",
            ));
            let end = if med_exp.is_finite() { last.add_seconds(med_exp) } else { last };
            out.push(FitsKeyword::new(
                "DATE-END",
                format!("'{}'", end.to_iso()),
                "last frame end (UTC)",
            ));
            continue;
        }
        let v0 = g[0].bare();
        if g.iter().all(|k| k.bare() == v0) {
            out.push(FitsKeyword {
                name: name.clone(),
                value: g[0].value.clone(),
                comment: g[0].comment.clone(),
            });
        }
    }
    out
}

fn norm_attrs(p: &XisfProperty) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> =
        p.attributes.iter().filter(|(k, _)| k != "id").cloned().collect();
    v.sort();
    v
}

/// `_aggregate_properties`: identical properties pass through untouched (base64
/// vectors stay valid); numeric properties that differ per frame collapse to the
/// median; `Observation:Time:Start/End` to the earliest/latest.  PCL:/PixInsight:
/// signatures are per-frame and are dropped.
pub fn aggregate_properties(metas: &[XisfMeta]) -> Vec<XisfProperty> {
    let n = metas.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<&XisfProperty>> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in metas {
        for p in &m.properties {
            match index.get(&p.id) {
                Some(&i) => groups[i].push(p),
                None => {
                    order.push(p.id.clone());
                    groups.push(vec![p]);
                    index.insert(p.id.clone(), groups.len() - 1);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (i, pid) in order.iter().enumerate() {
        let g = &groups[i];
        if g.len() != n || pid.starts_with("PCL:") || pid.starts_with("PixInsight:") {
            continue;
        }
        let same = g
            .iter()
            .all(|p| p.ty == g[0].ty && p.text == g[0].text && norm_attrs(p) == norm_attrs(g[0]));
        if same {
            out.push(g[0].clone());
            continue;
        }
        let ty = g[0].ty.clone();
        if matches!(
            ty.as_str(),
            "Float32"
                | "Float64"
                | "Int8"
                | "Int16"
                | "Int32"
                | "Int64"
                | "UInt8"
                | "UInt16"
                | "UInt32"
                | "UInt64"
        ) {
            let nums: Option<Vec<f64>> = g
                .iter()
                .map(|p| p.value().and_then(|v| v.trim().parse::<f64>().ok()))
                .collect();
            if let Some(nums) = nums {
                let med = median_f64(&nums);
                out.push(XisfProperty::value_prop(pid, &ty, &py_num(med)));
                continue;
            }
        }
        if ty == "TimePoint" {
            let vals: Vec<&str> = g.iter().filter_map(|p| p.value()).collect();
            if vals.len() == n {
                let pick = if pid.ends_with(":Start") {
                    vals.iter().min()
                } else if pid.ends_with(":End") {
                    vals.iter().max()
                } else {
                    None
                };
                if let Some(v) = pick {
                    out.push(XisfProperty::value_prop(pid, &ty, v));
                }
            }
        }
    }
    out
}

/// `_pws_props`: the fixed `PWS:` diagnostic set, in the order Python writes it.
pub fn pws_properties(diag: &PwsDiag) -> Vec<XisfProperty> {
    let mut out = vec![XisfProperty::text_prop(
        "PWS:Software",
        "String",
        "DWT · Partition-Weighted Stacking",
    )];
    fn num(out: &mut Vec<XisfProperty>, id: &str, ty: &str, v: Option<f64>) {
        if let Some(v) = v {
            out.push(XisfProperty::value_prop(id, ty, &py_num(v)));
        }
    }
    num(&mut out, "PWS:A", "Float64", diag.a);
    num(&mut out, "PWS:B", "Float64", diag.b);
    num(&mut out, "PWS:RejK", "Float64", diag.rej_k);
    num(&mut out, "PWS:RejGate", "Float64", diag.rej_gate);
    num(&mut out, "PWS:ScaleOut", "Float64", diag.scale_out);
    num(&mut out, "PWS:RejRate", "Float64", diag.rej_rate);
    if let Some(n) = diag.star_count {
        out.push(XisfProperty::value_prop("PWS:StarCount", "Int32", &n.to_string()));
    }
    num(&mut out, "PWS:FwhmFrameMedian", "Float64", diag.fwhm_frame_median);
    num(&mut out, "PWS:FwhmOutput", "Float64", diag.fwhm_output);
    num(&mut out, "PWS:NeffMedian", "Float64", diag.neff_median);
    num(&mut out, "PWS:SigmaFrameMedian", "Float64", diag.sigma_frame_median);
    num(&mut out, "PWS:SigmaSky", "Float64", diag.sigma_sky);
    out
}

/// Engine diagnostics for [`pws_properties`]; `None` = not available, and the
/// property is then simply not written (same as Python's `diag.get(k) is None`).
#[derive(Debug, Clone, Default)]
pub struct PwsDiag {
    pub a: Option<f64>,
    pub b: Option<f64>,
    pub rej_k: Option<f64>,
    pub rej_gate: Option<f64>,
    pub scale_out: Option<f64>,
    pub rej_rate: Option<f64>,
    pub star_count: Option<i64>,
    pub fwhm_frame_median: Option<f64>,
    pub fwhm_output: Option<f64>,
    pub neff_median: Option<f64>,
    pub sigma_frame_median: Option<f64>,
    pub sigma_sky: Option<f64>,
}

impl PwsDiag {
    /// The auto-cropped product is written by Python with a much smaller `diag`
    /// (`A`, `B`, `CropCover`, `CropBox`, `FwhmOutput` only), so its header
    /// carries just `Software`/`A`/`B`/`FwhmOutput` from `_pws_props`. Mirror
    /// that subset instead of leaking the main product's diagnostics into it.
    pub fn autocrop_subset(&self) -> Self {
        Self {
            a: self.a,
            b: self.b,
            fwhm_output: self.fwhm_output,
            ..Self::default()
        }
    }
}
