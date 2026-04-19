//! Model download from HuggingFace with progress display.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

// ========================================================================
// Model Registry
// ========================================================================

pub struct ModelInfo {
    pub name: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub description: &'static str,
}

pub const KNOWN_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "qwen3-asr-0.6b",
        repo: "Qwen/Qwen3-ASR-0.6B",
        files: &["model.safetensors", "vocab.json", "merges.txt"],
        description: "Qwen3-ASR 0.6B — fast, ~490 MB",
    },
    ModelInfo {
        name: "qwen3-asr-1.7b",
        repo: "Qwen/Qwen3-ASR-1.7B",
        files: &[
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vocab.json",
            "merges.txt",
        ],
        description: "Qwen3-ASR 1.7B — higher accuracy, ~3.4 GB",
    },
    ModelInfo {
        name: "qwen3-aligner-0.6b",
        repo: "Qwen/Qwen3-ASR-ForcedAligner-0.6B",
        files: &[
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vocab.json",
            "merges.txt",
        ],
        description: "Qwen3-ASR ForcedAligner 0.6B — word-level timestamps, ~1.6 GB",
    },
];

pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    let name_lower = name.to_lowercase();
    KNOWN_MODELS.iter().find(|m| m.name == name_lower)
}

// ========================================================================
// List Models
// ========================================================================

pub fn list_models() {
    eprintln!("Available models:\n");
    for m in KNOWN_MODELS {
        eprintln!("  {:<24} {}", m.name, m.description);
    }
    eprintln!();
    eprintln!("Usage: qwen-asr download <model-name> [--output <dir>]");
}

// ========================================================================
// Download
// ========================================================================

fn hf_url(repo: &str, file: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, file
    )
}

/// Format bytes as human-readable size.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Download a single file with progress display and resume support.
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let mut start_byte: u64 = 0;

    // Check for partial download (resume support)
    let part_path = dest.with_extension(
        dest.extension()
            .map(|e| format!("{}.part", e.to_string_lossy()))
            .unwrap_or_else(|| "part".to_string()),
    );
    if part_path.exists() {
        start_byte = fs::metadata(&part_path)
            .map(|m| m.len())
            .unwrap_or(0);
    }

    // Already fully downloaded?
    if dest.exists() {
        return Ok(());
    }

    // Build request
    let mut req = ureq::get(url);
    if start_byte > 0 {
        req = req.set("Range", &format!("bytes={}-", start_byte));
        eprint!(
            "  Resuming from {} ... ",
            format_bytes(start_byte)
        );
    }

    let resp = req.call().map_err(|e| format!("HTTP request failed: {}", e))?;

    // Parse content length
    let total_bytes = if start_byte > 0 {
        // For Range requests, Content-Range: bytes start-end/total
        resp.header("Content-Range")
            .and_then(|cr| cr.rsplit('/').next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    } else {
        resp.header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let mut reader = resp.into_reader();
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part_path)
        .map_err(|e| format!("Cannot open {}: {}", part_path.display(), e))?;

    let mut downloaded = start_byte;
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer
    let mut last_progress = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {}", e))?;
        downloaded += n as u64;

        // Update progress ~4 times per second
        let now = std::time::Instant::now();
        if now.duration_since(last_progress).as_millis() >= 250 || n == 0 {
            last_progress = now;
            let elapsed = now.duration_since(start_time).as_secs_f64();
            let speed = if elapsed > 0.0 {
                (downloaded - start_byte) as f64 / elapsed
            } else {
                0.0
            };

            if total_bytes > 0 {
                let pct = (downloaded as f64 / total_bytes as f64 * 100.0).min(100.0);
                eprint!(
                    "\r  {} / {} ({:.0}%) {}/s    ",
                    format_bytes(downloaded),
                    format_bytes(total_bytes),
                    pct,
                    format_bytes(speed as u64),
                );
            } else {
                eprint!(
                    "\r  {} downloaded, {}/s    ",
                    format_bytes(downloaded),
                    format_bytes(speed as u64),
                );
            }
        }
    }

    eprintln!(); // newline after progress

    // Rename .part to final destination
    fs::rename(&part_path, dest)
        .map_err(|e| format!("Cannot rename {} → {}: {}", part_path.display(), dest.display(), e))?;

    Ok(())
}

/// Download all files for a model.
pub fn download_model(model: &ModelInfo, output_dir: &str) -> Result<(), String> {
    let dir = Path::new(output_dir);
    fs::create_dir_all(dir)
        .map_err(|e| format!("Cannot create directory {}: {}", output_dir, e))?;

    let total_files = model.files.len();
    for (i, file_name) in model.files.iter().enumerate() {
        let dest = dir.join(file_name);
        if dest.exists() {
            eprintln!(
                "[{}/{}] {} — already exists, skipping",
                i + 1,
                total_files,
                file_name
            );
            continue;
        }

        let url = hf_url(model.repo, file_name);
        eprintln!("[{}/{}] Downloading {} ...", i + 1, total_files, file_name);
        download_file(&url, &dest)?;
    }

    eprintln!("\n✓ Model '{}' downloaded to {}", model.name, output_dir);
    Ok(())
}

// ========================================================================
// Interactive Prompt
// ========================================================================

/// Prompt user to download a model. Returns true if they accepted.
pub fn prompt_download(model_name: &str) -> bool {
    let model = match find_model(model_name) {
        Some(m) => m,
        None => return false,
    };

    eprintln!("Model directory '{}' not found.\n", model_name);
    eprintln!("  {} — {}\n", model.name, model.description);
    eprint!("Download now? [Y/n]: ");
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let answer = input.trim().to_lowercase();
    answer.is_empty() || answer == "y" || answer == "yes"
}

// ========================================================================
// CLI Entry Point
// ========================================================================

/// Handle the `download` subcommand. Returns true if handled (caller should exit).
pub fn handle_download_command(args: &[String]) -> bool {
    // Parse: download [--list] [<model-name>] [--output <dir>]
    let mut model_name: Option<String> = None;
    let mut output_dir: Option<String> = None;
    let mut show_list = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" | "-l" => {
                show_list = true;
            }
            "--output" | "-o" => {
                i += 1;
                output_dir = args.get(i).cloned();
            }
            "-h" | "--help" => {
                eprintln!("Usage: qwen-asr download [--list] [<model-name>] [--output <dir>]\n");
                eprintln!("Options:");
                eprintln!("  --list, -l       List available models");
                eprintln!("  --output, -o     Download directory (default: ./<model-name>/)");
                eprintln!("  -h, --help       Show this help");
                return true;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option for download: {}", other);
                    return true;
                }
                model_name = Some(other.to_string());
            }
        }
        i += 1;
    }

    if show_list || model_name.is_none() {
        list_models();
        return true;
    }

    let name = model_name.unwrap();
    let model = match find_model(&name) {
        Some(m) => m,
        None => {
            eprintln!("Unknown model: '{}'\n", name);
            list_models();
            std::process::exit(1);
        }
    };

    let dir = output_dir.unwrap_or_else(|| name.clone());
    match download_model(model, &dir) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("\nError: {}", e);
            std::process::exit(1);
        }
    }

    true
}
