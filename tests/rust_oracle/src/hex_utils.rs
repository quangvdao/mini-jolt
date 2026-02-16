pub(crate) fn hex_to_bytes(s: &str) -> Vec<u8> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        panic!("hex string must have even length");
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..s.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16).unwrap();
        let lo = (bytes[i + 1] as char).to_digit(16).unwrap();
        out.push(((hi << 4) | lo) as u8);
    }
    out
}

pub(crate) fn bytes_to_hex(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len() * 2);
    for x in b {
        out.push_str(&format!("{:02x}", x));
    }
    out
}
