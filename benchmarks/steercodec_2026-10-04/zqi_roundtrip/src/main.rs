//! `zqi_roundtrip <in.png> <out.png> <quality> [xyb|ycocg]`: zqi lossy encode at `quality`, decode, write the decoded RGB8
//! PNG, print `bytes=<encoded size>`. Default color: zqi's own default (XYB). Used by the steercodec steering benchmark.
use std::fs::File;
use std::io::BufWriter;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() >= 4, "usage: zqi_roundtrip <in.png> <out.png> <quality> [xyb|ycocg]");
    let dec = png::Decoder::new(std::io::BufReader::new(File::open(&a[1]).expect("open input")));
    let mut reader = dec.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert!(
        info.color_type == png::ColorType::Rgb && info.bit_depth == png::BitDepth::Eight,
        "8-bit RGB PNG required"
    );
    let (w, h) = (info.width, info.height);
    let pixels = &buf[..info.buffer_size()];
    let mut cfg = zqi::LossyConfig::new(a[3].parse().expect("quality"));
    if let Some(c) = a.get(4) {
        cfg = cfg.with_color(match c.as_str() {
            "xyb" => zqi::LossyColor::Xyb,
            "ycocg" => zqi::LossyColor::YCoCg,
            other => panic!("unknown color {other}"),
        });
    }
    let zinfo = zqi::ImageInfo::new(w, h, zqi::Channels::Rgb, zqi::SampleType::U8);
    let enc = zqi::encode_lossy(pixels, w as usize * 3, &zinfo, &zqi::Metadata::default(), &cfg).expect("encode");
    let (dinfo, out) = zqi::decode(&enc).expect("decode");
    assert_eq!((dinfo.width, dinfo.height), (w, h));
    let mut e = png::Encoder::new(BufWriter::new(File::create(&a[2]).expect("create output")), w, h);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().expect("png write").write_image_data(&out).expect("png data");
    println!("bytes={}", enc.len());
}
