//! Minimal XISF 1.0 reader/writer: single-image, single-channel, planar Float32
//! (the subset the PWS stack needs). Byte-compatible with `xisf_io.py` for the
//! uncompressed layout produced by PixInsight/WBPP and by this tool.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::image::Img;

#[derive(Debug, Clone)]
pub struct XisfMeta {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub sample_format: String,
    pub color_space: String,
    pub pixel_storage: String,
    pub bounds: Option<(f32, f32)>,
    /// Raw FITSKeyword `name -> value` (quotes kept as written).
    pub keywords: Vec<(String, String)>,
    /// XISF `<Property id="..." value="..."/>` pairs (the CFA pattern lives here
    /// on PixInsight files that do not carry a FITS `BAYERPAT` keyword).
    pub properties: Vec<(String, String)>,
}

impl XisfMeta {
    pub fn keyword(&self, name: &str) -> Option<&str> {
        self.keywords
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
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

    // FITSKeyword children: name/value pairs
    let mut keywords = Vec::new();
    let mut rest = image_tag.to_string();
    let mut cursor = image_end;
    while let Some(p) = xml[cursor..].find("<FITSKeyword") {
        let s = cursor + p;
        let e = s + xml[s..].find('>').unwrap_or(0);
        let tag = &xml[s..e];
        if let (Some(n), Some(v)) = (attr(tag, "name"), attr(tag, "value")) {
            keywords.push((n.to_string(), v.to_string()));
        }
        cursor = e;
    }
    let _ = &mut rest;

    // XISF <Property> elements anywhere in the header (PCL:CFASourcePattern etc.)
    let mut properties = Vec::new();
    let mut cursor = 0usize;
    while let Some(p) = xml[cursor..].find("<Property") {
        let s = cursor + p;
        let e = s + xml[s..].find('>').unwrap_or(0);
        let tag = &xml[s..e];
        if let (Some(id), Some(v)) = (attr(tag, "id"), attr(tag, "value")) {
            properties.push((id.to_string(), v.to_string()));
        }
        cursor = e;
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

/// Values written into the output header. Mirrors `save_linear_xisf` in `core/pws.py`.
pub struct OutMeta {
    pub total_exp: f64,
    pub ncombine: u32,
    pub stack_mode: String,
    pub passthrough: Vec<(String, String)>,
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
    let mut push_kw = |name: &str, value: String, comment: &str| {
        kws.push_str(&format!(
            "      <FITSKeyword name=\"{name}\" value=\"{value}\" comment=\"{comment}\"/>\n"
        ));
    };
    push_kw("TOTALEXP", format!("{:.1}", meta.total_exp), "total exposure seconds of combined frames");
    push_kw("NCOMBINE", meta.ncombine.to_string(), "frames combined");
    push_kw("STACKMODE", format!("'{}\'", meta.stack_mode), "partition-weighted stacking (linear output, no stretch)");
    // pass through the first frame's instrument/observation keywords
    for (k, v) in &meta.passthrough {
        if matches!(k.as_str(), "TOTALEXP" | "NCOMBINE" | "STACKMODE") {
            continue;
        }
        let esc = v.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;");
        kws.push_str(&format!("      <FITSKeyword name=\"{k}\" value=\"{esc}\"/>\n"));
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
