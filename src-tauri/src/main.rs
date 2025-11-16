// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
struct PlayData {
    time: f64,
    players: Vec<Player>,
    ball: BallPosition,
}

#[derive(Debug, Serialize, Deserialize)]
struct Player {
    id: String,
    x: f64,
    y: f64,
    role: String,
    possession: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct BallPosition {
    x: f64,
    y: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LabelData {
    play_id: String,
    outcome: String,
    tags: Vec<String>,
    timestamp: String,
}

// Load play data from file
#[tauri::command]
fn load_play_data(path: String) -> Result<PlayData, String> {
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let play_data: PlayData = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(play_data)
}

// Save label to file
#[tauri::command]
fn save_label(label: LabelData, output_path: String) -> Result<String, String> {
    // Load existing labels
    let mut labels: Vec<LabelData> = if PathBuf::from(&output_path).exists() {
        let content = fs::read_to_string(&output_path)
            .map_err(|e| format!("Failed to read labels file: {}", e))?;
        serde_json::from_str(&content).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    };

    // Add new label
    labels.push(label);

    // Save to file
    let json = serde_json::to_string_pretty(&labels)
        .map_err(|e| format!("Failed to serialize labels: {}", e))?;

    fs::write(&output_path, json)
        .map_err(|e| format!("Failed to write labels file: {}", e))?;

    Ok(format!("Label saved to {}", output_path))
}

// Get app data directory
#[tauri::command]
fn get_app_data_dir(app_handle: tauri::AppHandle) -> Result<String, String> {
    app_handle
        .path_resolver()
        .app_data_dir()
        .map(|p| p.to_string_lossy().to_string())
        .ok_or_else(|| "Failed to get app data directory".to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            load_play_data,
            save_label,
            get_app_data_dir
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
