//! GPT-2 byte-level BPE tokenizer for Qwen.

use std::collections::HashMap;

// GPT-2 bytes-to-unicode mapping
fn init_gpt2_mapping() -> ([i32; 256], [i32; 512]) {
    let mut byte_to_unicode = [0i32; 256];
    let mut unicode_to_byte = [-1i32; 512];

    let mut n = 0i32;
    for b in 0..256i32 {
        let is_normal = (33..=126).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b);

        if is_normal {
            byte_to_unicode[b as usize] = b;
        } else {
            byte_to_unicode[b as usize] = 256 + n;
            n += 1;
        }
    }

    for (b, &bu) in byte_to_unicode.iter().enumerate() {
        let cp = bu as usize;
        if cp < 512 {
            unicode_to_byte[cp] = b as i32;
        }
    }

    (byte_to_unicode, unicode_to_byte)
}

fn utf8_encode_cp(cp: u32) -> Vec<u8> {
    if cp < 0x80 {
        vec![cp as u8]
    } else if cp < 0x800 {
        vec![
            (0xC0 | (cp >> 6)) as u8,
            (0x80 | (cp & 0x3F)) as u8,
        ]
    } else {
        vec![
            (0xE0 | (cp >> 12)) as u8,
            (0x80 | ((cp >> 6) & 0x3F)) as u8,
            (0x80 | (cp & 0x3F)) as u8,
        ]
    }
}

/// Decode a GPT-2 encoded token string (vocab key) to raw bytes.
/// Returns raw bytes instead of String because a single BPE token may
/// represent only a partial UTF-8 sequence (e.g. 2 of 3 bytes for a CJK
/// character).  The caller must accumulate bytes from multiple tokens and
/// convert to UTF-8 only after the full sequence is available.
fn decode_gpt2_token_bytes(token_str: &str, unicode_to_byte: &[i32; 512]) -> Vec<u8> {
    let mut bytes = Vec::new();

    for ch in token_str.chars() {
        let cp = ch as u32;
        if cp < 512 && unicode_to_byte[cp as usize] >= 0 {
            bytes.push(unicode_to_byte[cp as usize] as u8);
        } else {
            bytes.push(b'?');
        }
    }

    bytes
}

/// Convert UTF-8 bytes to GPT-2 byte-level unicode string.
fn text_to_bpe_unicode(text: &str, byte_to_unicode: &[i32; 256]) -> String {
    let mut out = String::new();
    for &b in text.as_bytes() {
        let cp = byte_to_unicode[b as usize] as u32;
        for byte in utf8_encode_cp(cp) {
            out.push(byte as char);
        }
    }
    // Actually, we need to push the encoded codepoint as a char
    let mut out2 = String::new();
    for &b in text.as_bytes() {
        let cp = byte_to_unicode[b as usize] as u32;
        if let Some(ch) = char::from_u32(cp) {
            out2.push(ch);
        }
    }
    out2
}

fn utf8_char_len(c: u8) -> usize {
    if c & 0x80 == 0 { 1 }
    else if c & 0xE0 == 0xC0 { 2 }
    else if c & 0xF0 == 0xE0 { 3 }
    else if c & 0xF8 == 0xF0 { 4 }
    else { 1 }
}

fn split_utf8_symbols(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut syms = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let len = utf8_char_len(bytes[i]);
        let end = (i + len).min(bytes.len());
        if let Ok(ch) = std::str::from_utf8(&bytes[i..end]) {
            syms.push(ch.to_string());
        }
        i = end;
    }
    syms
}

pub struct QwenTokenizer {
    pub vocab_size: usize,
    id_to_text: Vec<Option<String>>,
    id_to_bytes: Vec<Option<Vec<u8>>>,
    #[allow(dead_code)]
    id_to_bpe: Vec<Option<String>>,
    vocab_map: HashMap<String, i32>,
    merge_map: HashMap<String, i32>,
    byte_to_unicode: [i32; 256],
    #[allow(dead_code)]
    unicode_to_byte: [i32; 512],
}

impl QwenTokenizer {
    pub fn load(vocab_json_path: &str) -> Option<Self> {
        let (byte_to_unicode, unicode_to_byte) = init_gpt2_mapping();

        // Read vocab.json
        let json = std::fs::read_to_string(vocab_json_path).ok()?;

        // Parse vocab.json: { "token": id, ... }
        let mut max_id = 0i32;
        let mut entries: Vec<(String, i32)> = Vec::new();

        let bytes = json.as_bytes();
        let mut pos = 0;
        skip_ws(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b'{' {
            return None;
        }
        pos += 1;

        loop {
            skip_ws(bytes, &mut pos);
            if pos >= bytes.len() || bytes[pos] == b'}' {
                break;
            }
            if bytes[pos] == b',' {
                pos += 1;
                continue;
            }

            let key = parse_json_string_tok(bytes, &mut pos)?;
            skip_ws(bytes, &mut pos);
            if pos >= bytes.len() || bytes[pos] != b':' {
                return None;
            }
            pos += 1;
            let id = parse_json_int_tok(bytes, &mut pos)? as i32;

            if id > max_id {
                max_id = id;
            }
            entries.push((key, id));
        }

        let vocab_size = (max_id + 1) as usize;
        let mut id_to_text = vec![None; vocab_size];
        let mut id_to_bytes: Vec<Option<Vec<u8>>> = vec![None; vocab_size];
        let mut id_to_bpe = vec![None; vocab_size];
        let mut vocab_map = HashMap::new();

        for (key, id) in entries {
            let idx = id as usize;
            if idx < vocab_size {
                let raw_bytes = decode_gpt2_token_bytes(&key, &unicode_to_byte);
                // id_to_text: lossy UTF-8 for display/legacy use
                let text = String::from_utf8_lossy(&raw_bytes).into_owned();
                id_to_text[idx] = Some(text);
                id_to_bytes[idx] = Some(raw_bytes);
                vocab_map.insert(key.clone(), id);
                id_to_bpe[idx] = Some(key);
            }
        }

        // Load merges.txt
        let merge_map = load_merges(vocab_json_path);

        Some(QwenTokenizer {
            vocab_size,
            id_to_text,
            id_to_bytes,
            id_to_bpe,
            vocab_map,
            merge_map,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn decode(&self, token_id: i32) -> &str {
        if token_id < 0 || token_id as usize >= self.vocab_size {
            return "";
        }
        match &self.id_to_text[token_id as usize] {
            Some(s) => s.as_str(),
            None => "",
        }
    }

    /// Decode a token to its raw bytes. Unlike `decode()`, this preserves
    /// partial UTF-8 sequences so that the caller can accumulate bytes from
    /// multiple tokens before converting to a valid UTF-8 string.
    pub fn decode_bytes(&self, token_id: i32) -> &[u8] {
        if token_id < 0 || token_id as usize >= self.vocab_size {
            return b"";
        }
        match &self.id_to_bytes[token_id as usize] {
            Some(b) => b.as_slice(),
            None => b"",
        }
    }

    pub fn encode(&self, text: &str) -> Option<Vec<i32>> {
        if text.is_empty() {
            return None;
        }

        let mapped = text_to_bpe_unicode(text, &self.byte_to_unicode);
        let ids = self.encode_bpe_word(&mapped)?;
        Some(ids)
    }

    fn encode_bpe_word(&self, mapped: &str) -> Option<Vec<i32>> {
        if mapped.is_empty() {
            return Some(Vec::new());
        }

        let mut syms = split_utf8_symbols(mapped);
        if syms.is_empty() {
            return Some(Vec::new());
        }

        while syms.len() > 1 {
            let mut best_rank = i32::MAX;
            let mut best_i = -1i32;

            for i in 0..syms.len() - 1 {
                let pair = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(&rank) = self.merge_map.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i as i32;
                    }
                }
            }

            if best_i < 0 || best_rank == i32::MAX {
                break;
            }

            let i = best_i as usize;
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }

        let mut ids = Vec::new();
        for sym in &syms {
            let id = self.vocab_map.get(sym.as_str()).copied()?;
            ids.push(id);
        }

        Some(ids)
    }
}

fn load_merges(vocab_path: &str) -> HashMap<String, i32> {
    let mut merge_map = HashMap::new();

    // Derive merges.txt path from vocab.json path
    let merges_path = if let Some(slash) = vocab_path.rfind('/') {
        format!("{}/merges.txt", &vocab_path[..slash])
    } else {
        "merges.txt".to_string()
    };

    let content = match std::fs::read_to_string(&merges_path) {
        Ok(c) => c,
        Err(_) => return merge_map,
    };

    let mut rank = 0i32;
    for line in content.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(space_pos) = line.find(' ') {
            let a = &line[..space_pos];
            let b = line[space_pos + 1..].trim_start();
            if !a.is_empty() && !b.is_empty() {
                let key = format!("{} {}", a, b);
                merge_map.insert(key, rank);
                rank += 1;
            }
        }
    }

    merge_map
}

// Minimal JSON parsing helpers
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() {
        match bytes[*pos] {
            b' ' | b'\n' | b'\r' | b'\t' => *pos += 1,
            _ => break,
        }
    }
}

fn parse_json_string_tok(bytes: &[u8], pos: &mut usize) -> Option<String> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    let mut result = Vec::new();
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        if bytes[*pos] == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return None;
            }
            match bytes[*pos] {
                b'n' => result.push(b'\n'),
                b't' => result.push(b'\t'),
                b'"' => result.push(b'"'),
                b'\\' => result.push(b'\\'),
                b'/' => result.push(b'/'),
                b'u' => {
                    *pos += 1;
                    let mut cp = 0u32;
                    for _ in 0..4 {
                        if *pos >= bytes.len() {
                            return None;
                        }
                        cp <<= 4;
                        let c = bytes[*pos];
                        cp |= match c {
                            b'0'..=b'9' => (c - b'0') as u32,
                            b'a'..=b'f' => (c - b'a' + 10) as u32,
                            b'A'..=b'F' => (c - b'A' + 10) as u32,
                            _ => return None,
                        };
                        *pos += 1;
                    }
                    if let Some(ch) = char::from_u32(cp) {
                        let mut buf = [0u8; 4];
                        let s = ch.encode_utf8(&mut buf);
                        result.extend_from_slice(s.as_bytes());
                    }
                    continue;
                }
                other => result.push(other),
            }
        } else {
            result.push(bytes[*pos]);
        }
        *pos += 1;
    }

    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    String::from_utf8(result).ok()
}

fn parse_json_int_tok(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    skip_ws(bytes, pos);
    let mut neg = false;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        neg = true;
        *pos += 1;
    }
    let mut val: i64 = 0;
    let mut found = false;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        val = val * 10 + (bytes[*pos] - b'0') as i64;
        *pos += 1;
        found = true;
    }
    if !found {
        return None;
    }
    Some(if neg { -val } else { val })
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert raw bytes to a GPT-2 BPE token string using byte→unicode mapping.
    fn bytes_to_gpt2_token(raw_bytes: &[u8], byte_to_unicode: &[i32; 256]) -> String {
        let mut s = String::new();
        for &b in raw_bytes {
            if let Some(ch) = char::from_u32(byte_to_unicode[b as usize] as u32) {
                s.push(ch);
            }
        }
        s
    }

    // ----------------------------------------------------------------
    // decode_gpt2_token_bytes: round-trip correctness
    // ----------------------------------------------------------------

    #[test]
    fn test_roundtrip_ascii() {
        let (btu, utb) = init_gpt2_mapping();
        let original = b"hello";
        let token_str = bytes_to_gpt2_token(original, &btu);
        let decoded = decode_gpt2_token_bytes(&token_str, &utb);
        assert_eq!(decoded, original);
        assert_eq!(String::from_utf8(decoded).unwrap(), "hello");
    }

    #[test]
    fn test_roundtrip_cjk_full_char() {
        let (btu, utb) = init_gpt2_mapping();
        // "地" = UTF-8 [0xE5, 0x9C, 0xB0]
        let original = "地".as_bytes();
        let token_str = bytes_to_gpt2_token(original, &btu);
        let decoded = decode_gpt2_token_bytes(&token_str, &utb);
        assert_eq!(decoded, original);
        assert_eq!(String::from_utf8(decoded).unwrap(), "地");
    }

    // ----------------------------------------------------------------
    // Core regression test: split UTF-8 across two BPE tokens
    // ----------------------------------------------------------------

    #[test]
    fn test_decode_bytes_split_utf8_cjk() {
        // "地" = UTF-8 [0xE5, 0x9C, 0xB0]
        // Simulate BPE splitting into two tokens:
        //   Token A covers bytes [0xE5, 0x9C]  (first 2 of 3)
        //   Token B covers byte  [0xB0]         (last 1 of 3)
        let (btu, utb) = init_gpt2_mapping();

        let part1_bytes: &[u8] = &[0xE5, 0x9C];
        let part2_bytes: &[u8] = &[0xB0];

        let token_a = bytes_to_gpt2_token(part1_bytes, &btu);
        let token_b = bytes_to_gpt2_token(part2_bytes, &btu);

        let decoded_a = decode_gpt2_token_bytes(&token_a, &utb);
        let decoded_b = decode_gpt2_token_bytes(&token_b, &utb);

        // Each part alone is NOT valid UTF-8
        assert!(String::from_utf8(decoded_a.clone()).is_err(),
            "Part 1 alone should NOT be valid UTF-8");
        assert!(String::from_utf8(decoded_b.clone()).is_err(),
            "Part 2 alone should NOT be valid UTF-8");

        // But concatenated they form valid UTF-8 for "地"
        let mut combined = decoded_a;
        combined.extend_from_slice(&decoded_b);
        assert_eq!(combined, vec![0xE5, 0x9C, 0xB0]);
        assert_eq!(String::from_utf8(combined).unwrap(), "地");
    }

    #[test]
    fn test_decode_bytes_split_utf8_2byte_char() {
        // "é" = UTF-8 [0xC3, 0xA9]
        // Simulate BPE splitting each byte into its own token
        let (btu, utb) = init_gpt2_mapping();

        let part1: &[u8] = &[0xC3];
        let part2: &[u8] = &[0xA9];

        let decoded_1 = decode_gpt2_token_bytes(
            &bytes_to_gpt2_token(part1, &btu), &utb);
        let decoded_2 = decode_gpt2_token_bytes(
            &bytes_to_gpt2_token(part2, &btu), &utb);

        assert!(String::from_utf8(decoded_1.clone()).is_err());
        assert!(String::from_utf8(decoded_2.clone()).is_err());

        let mut combined = decoded_1;
        combined.extend_from_slice(&decoded_2);
        assert_eq!(String::from_utf8(combined).unwrap(), "é");
    }

    #[test]
    fn test_decode_bytes_split_utf8_4byte_emoji() {
        // "🦀" = UTF-8 [0xF0, 0x9F, 0xA6, 0x80]
        // Simulate BPE splitting into two halves
        let (btu, utb) = init_gpt2_mapping();

        let part1: &[u8] = &[0xF0, 0x9F];
        let part2: &[u8] = &[0xA6, 0x80];

        let decoded_1 = decode_gpt2_token_bytes(
            &bytes_to_gpt2_token(part1, &btu), &utb);
        let decoded_2 = decode_gpt2_token_bytes(
            &bytes_to_gpt2_token(part2, &btu), &utb);

        assert!(String::from_utf8(decoded_1.clone()).is_err());
        assert!(String::from_utf8(decoded_2.clone()).is_err());

        let mut combined = decoded_1;
        combined.extend_from_slice(&decoded_2);
        assert_eq!(String::from_utf8(combined).unwrap(), "🦀");
    }

    // ----------------------------------------------------------------
    // Mixed content: ASCII + Emoji + CJK
    // ----------------------------------------------------------------

    #[test]
    fn test_decode_bytes_mixed_content_rust_crab_chinese() {
        let (btu, utb) = init_gpt2_mapping();

        let full_text = "Rust🦀真棒";
        let all_bytes = full_text.as_bytes();

        // Decode each byte as its own token, accumulate
        let mut accumulated = Vec::new();
        for &b in all_bytes {
            let token_str = bytes_to_gpt2_token(&[b], &btu);
            let decoded = decode_gpt2_token_bytes(&token_str, &utb);
            accumulated.extend_from_slice(&decoded);
        }

        assert_eq!(accumulated, all_bytes);
        assert_eq!(String::from_utf8(accumulated).unwrap(), "Rust🦀真棒");
    }

    // ----------------------------------------------------------------
    // Regression: ensure old lossy path would have corrupted
    // ----------------------------------------------------------------

    #[test]
    fn test_lossy_corruption_proof() {
        // Demonstrate that per-token String::from_utf8_lossy WOULD corrupt,
        // while byte-level accumulation does NOT.
        let (btu, utb) = init_gpt2_mapping();

        // "地址" = [E5 9C B0] [E5 9D 80]
        // Split: token_a=[E5,9C], token_b=[B0,E5], token_c=[9D,80]
        let splits: &[&[u8]] = &[
            &[0xE5, 0x9C],
            &[0xB0, 0xE5],
            &[0x9D, 0x80],
        ];

        // Old approach: decode each token to String independently (lossy)
        let mut lossy_result = String::new();
        for &part in splits {
            let token_str = bytes_to_gpt2_token(part, &btu);
            let decoded = decode_gpt2_token_bytes(&token_str, &utb);
            lossy_result.push_str(&String::from_utf8_lossy(&decoded));
        }
        // Lossy approach produces replacement characters
        assert!(lossy_result.contains('\u{FFFD}'),
            "Lossy per-token decode SHOULD produce U+FFFD, got: {:?}", lossy_result);
        assert_ne!(lossy_result, "地址");

        // New approach: accumulate bytes, convert once
        let mut byte_buf = Vec::new();
        for &part in splits {
            let token_str = bytes_to_gpt2_token(part, &btu);
            let decoded = decode_gpt2_token_bytes(&token_str, &utb);
            byte_buf.extend_from_slice(&decoded);
        }
        let correct_result = String::from_utf8(byte_buf).unwrap();
        assert_eq!(correct_result, "地址");
        assert!(!correct_result.contains('\u{FFFD}'));
    }

    // ----------------------------------------------------------------
    // GPT-2 mapping correctness
    // ----------------------------------------------------------------

    #[test]
    fn test_gpt2_mapping_all_256_bytes_roundtrip() {
        let (btu, utb) = init_gpt2_mapping();

        // Every byte value 0..255 must survive a round-trip
        for b in 0u8..=255 {
            let cp = btu[b as usize];
            assert!(cp >= 0 && cp < 512,
                "byte {:#04x} mapped to out-of-range codepoint {}", b, cp);

            let recovered = utb[cp as usize];
            assert_eq!(recovered, b as i32,
                "byte {:#04x} → cp {} → byte {:#04x} (expected {:#04x})",
                b, cp, recovered, b);
        }
    }

    #[test]
    fn test_gpt2_mapping_is_bijective() {
        let (btu, _utb) = init_gpt2_mapping();

        // All 256 codepoints must be distinct
        let mut seen = std::collections::HashSet::new();
        for b in 0..256 {
            let cp = btu[b];
            assert!(seen.insert(cp),
                "byte {} and another byte both map to codepoint {}", b, cp);
        }
        assert_eq!(seen.len(), 256);
    }
}
