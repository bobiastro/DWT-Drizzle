use std::path::PathBuf;

use dwt_backend::wbpp::{self, WbppCommand};

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "dwt-backend".to_string());
    let Some(command) = args.next() else {
        return Err(usage(&program).into());
    };
    match command.as_str() {
        "wbpp-scan" => {
            let directory = required_path(args.next(), "WBPP output directory")?;
            let report = wbpp::scan_wbpp_dir(&directory)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        "wbpp-plan" => {
            let directory = required_path(args.next(), "WBPP output directory")?;
            let plan = wbpp::plan_from_wbpp(&directory)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
            Ok(())
        }
        "wbpp-request" => {
            let directory = required_path(args.next(), "WBPP output directory")?;
            let out = required_path(args.next(), "request output file")?;
            let request = wbpp::request_from_wbpp(&directory)?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(out, serde_json::to_vec_pretty(&request)?)?;
            Ok(())
        }
        "wbpp-validate" => {
            let request_path = required_path(args.next(), "request JSON file")?;
            let request: WbppCommand = serde_json::from_slice(&std::fs::read(request_path)?)?;
            wbpp::validate_request(&request)?;
            println!("valid");
            Ok(())
        }
        "wbpp-pws" => {
            let request_path = required_path(args.next(), "request JSON file")?;
            let request: WbppCommand = serde_json::from_slice(&std::fs::read(request_path)?)?;
            let mut last_progress = None;
            let mut sequence = 0;
            let result = wbpp::run_calibrated_pws_with_progress(&request, |phase, done, total| {
                let bucket = done.saturating_mul(10) / total.max(1);
                if done == 0 || last_progress != Some((phase, bucket)) {
                    sequence += 1;
                    eprintln!("[DWT] {sequence} {phase}: {done}/{total}");
                    last_progress = Some((phase, bucket));
                }
            })?;
            let result_path = required_path(args.next(), "result JSON file")?;
            if let Some(parent) = result_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(result_path, serde_json::to_vec_pretty(&result)?)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("{}", usage(&program));
            Ok(())
        }
        _ => Err(usage(&program).into()),
    }
}

fn required_path(
    value: Option<String>,
    label: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let Some(value) = value else {
        return Err(format!("missing {label}").into());
    };
    Ok(PathBuf::from(value))
}

fn usage(program: &str) -> String {
    format!(
        "Usage:\n\
         {program} wbpp-scan <WBPP-output-dir>\n\
         {program} wbpp-plan <WBPP-output-dir>\n\
         {program} wbpp-request <WBPP-output-dir> <request.json>\n\
         {program} wbpp-validate <request.json>\n\
         {program} wbpp-pws <request.json> <result.json>\n\
         {program} help"
    )
}
