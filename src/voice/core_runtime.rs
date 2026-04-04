use std::{
    collections::HashMap,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    time::Duration,
};

use flate2::read::GzDecoder;
use log::{info, warn};
use once_cell::sync::OnceCell;
use reqwest::{
    Client as HttpClient,
    header::{ACCEPT, AUTHORIZATION, USER_AGENT},
};
use serde::Deserialize;
use voicevox_core::{
    AccelerationMode, AudioQuery, CharacterMeta, StyleId,
    nonblocking::{Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile},
};

use super::types::VoiceCoreConfig;

const OPEN_JTALK_REPO: &str = "r9y9/open_jtalk";
const OPEN_JTALK_TAG: &str = "v1.11.1";
const OPEN_JTALK_ASSET_NAME: &str = "open_jtalk_dic_utf_8-1.11.tar.gz";

const VOICEVOX_VVM_REPO: &str = "VOICEVOX/voicevox_vvm";
const VOICEVOX_ONNXRUNTIME_REPO: &str = "VOICEVOX/onnxruntime-builder";
const ONNXRUNTIME_RELEASE_TAG_PREFIX: &str = "voicevox_onnxruntime";
const ONNXRUNTIME_ASSET_PREFIX: &str = "voicevox_";
const MODELS_DIR_NAME: &str = "vvms";
const MODELS_TERMS_FILE: &str = "TERMS.txt";
const MODELS_README_FILE: &str = "README.txt";
const SUPPORTED_VVM_MAJOR: u64 = 0;
const SUPPORTED_VVM_MINOR: u64 = 16;
const VVM_RELEASE_MARKER_FILE: &str = ".nelfie_vvm_release_tag";
const VVM_DOWNLOAD_CONCURRENCY: usize = 4;

static GITHUB_HTTP_CLIENT: OnceCell<HttpClient> = OnceCell::new();

#[derive(Debug, Deserialize)]
struct GithubRelease {
    tag_name: String,
    assets: Vec<GithubAsset>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
}

#[derive(Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug)]
struct DiscoveredModel {
    model_path: PathBuf,
    voice_model: VoiceModelFile,
    style_ids: Vec<u32>,
    version_key: String,
}

struct VoiceVoxInitDownloader;

impl VoiceVoxInitDownloader {
    async fn prepare(config: &VoiceCoreConfig) -> Result<(), String> {
        let dict_dir = PathBuf::from(&config.open_jtalk_dict_dir);
        let vvm_dir = PathBuf::from(&config.vvm_dir);

        // 起動待ち時間を短縮するため、独立したアセット準備を並列化する。
        tokio::try_join!(
            Self::ensure_onnxruntime_library(config),
            CoreRuntime::ensure_open_jtalk_dictionary(&dict_dir),
            CoreRuntime::ensure_vvm_models(&vvm_dir),
        )?;

        if vvm_dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| !name.eq_ignore_ascii_case(MODELS_DIR_NAME))
            .unwrap_or(false)
        {
            warn!(
                "VVM directory '{}' does not end with '{}'; ensure this is intentional",
                vvm_dir.display(),
                MODELS_DIR_NAME
            );
        }

        Ok(())
    }

    async fn ensure_onnxruntime_library(config: &VoiceCoreConfig) -> Result<(), String> {
        if !config.onnxruntime_filename.trim().is_empty() {
            info!("VOICEVOX_ONNXRUNTIME_FILENAME is set; skipping ONNX Runtime auto-download");
            return Ok(());
        }

        if Self::has_preferred_onnxruntime_library() {
            return Ok(());
        }

        if Self::sync_auto_discovered_onnxruntime_into_preferred_dir()? {
            return Ok(());
        }

        let release_tag = Self::onnxruntime_release_tag();
        let asset_name = Self::onnxruntime_asset_name()?;

        info!(
            "ONNX Runtime library is missing; downloading '{}' from release '{}'",
            asset_name, release_tag
        );

        let release = CoreRuntime::fetch_release_by_tag(VOICEVOX_ONNXRUNTIME_REPO, &release_tag)
            .await
            .map_err(|e| {
                format!(
                    "failed to fetch ONNX Runtime release '{}' from '{}': {}",
                    release_tag, VOICEVOX_ONNXRUNTIME_REPO, e
                )
            })?;

        let asset = release
            .assets
            .iter()
            .find(|asset| asset.name == asset_name)
            .ok_or_else(|| {
                format!(
                    "ONNX Runtime asset '{}' was not found in release '{}' of '{}'",
                    asset_name, release.tag_name, VOICEVOX_ONNXRUNTIME_REPO
                )
            })?;

        let archive = CoreRuntime::download_asset_bytes(&asset.browser_download_url).await?;
        let output_dir = Self::choose_onnxruntime_download_dir()?;
        let extracted = Self::extract_onnxruntime_libraries(&archive, &output_dir)?;

        if extracted == 0 {
            return Err(format!(
                "ONNX Runtime archive '{}' did not contain '{}'",
                asset_name,
                Onnxruntime::LIB_VERSIONED_FILENAME
            ));
        }

        if !Self::has_preferred_onnxruntime_library() {
            return Err(format!(
                "ONNX Runtime download completed, but '{}' is still unavailable in preferred paths",
                Onnxruntime::LIB_VERSIONED_FILENAME
            ));
        }

        info!(
            "ONNX Runtime downloaded successfully (tag={}, files={}, output_dir={})",
            release.tag_name,
            extracted,
            output_dir.display()
        );

        Ok(())
    }

    fn has_preferred_onnxruntime_library() -> bool {
        Self::preferred_onnxruntime_download_dirs()
            .into_iter()
            .any(|dir| {
                dir.join(Onnxruntime::LIB_VERSIONED_FILENAME).is_file()
                    || dir.join(Onnxruntime::LIB_UNVERSIONED_FILENAME).is_file()
            })
    }

    fn sync_auto_discovered_onnxruntime_into_preferred_dir() -> Result<bool, String> {
        let Some(source_dir) = CoreRuntime::onnxruntime_search_dirs()
            .into_iter()
            .find(|dir| {
                dir.join(Onnxruntime::LIB_VERSIONED_FILENAME).is_file()
                    || dir.join(Onnxruntime::LIB_UNVERSIONED_FILENAME).is_file()
            })
        else {
            return Ok(false);
        };

        let output_dir = Self::choose_onnxruntime_download_dir()?;
        let mut copied = 0usize;

        for filename in [
            Onnxruntime::LIB_VERSIONED_FILENAME,
            Onnxruntime::LIB_UNVERSIONED_FILENAME,
        ] {
            let src = source_dir.join(filename);
            if !src.is_file() {
                continue;
            }

            let dst = output_dir.join(filename);
            if src == dst {
                continue;
            }

            std::fs::copy(&src, &dst).map_err(|e| {
                format!(
                    "failed to copy ONNX Runtime file from '{}' to '{}': {}",
                    src.display(),
                    dst.display(),
                    e
                )
            })?;
            copied += 1;
        }

        if copied > 0 {
            info!(
                "ONNX Runtime was found in '{}' and copied to preferred directory '{}'",
                source_dir.display(),
                output_dir.display()
            );
        }

        Ok(Self::has_preferred_onnxruntime_library())
    }

    fn onnxruntime_release_tag() -> String {
        format!(
            "{}-{}",
            ONNXRUNTIME_RELEASE_TAG_PREFIX,
            Onnxruntime::LIB_VERSION.trim()
        )
    }

    fn onnxruntime_asset_name() -> Result<String, String> {
        let artifact = Self::onnxruntime_artifact_name()?;
        let version = Onnxruntime::LIB_VERSION.trim();
        Ok(format!(
            "{}{}-{}.tgz",
            ONNXRUNTIME_ASSET_PREFIX, artifact, version
        ))
    }

    fn onnxruntime_artifact_name() -> Result<&'static str, String> {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("windows", "x86_64") => Ok("onnxruntime-win-x64"),
            ("windows", "x86") => Ok("onnxruntime-win-x86"),
            ("linux", "x86_64") => Ok("onnxruntime-linux-x64"),
            ("linux", "aarch64") => Ok("onnxruntime-linux-arm64"),
            ("linux", "arm") => Ok("onnxruntime-linux-armhf"),
            ("macos", "x86_64") => Ok("onnxruntime-osx-x86_64"),
            ("macos", "aarch64") => Ok("onnxruntime-osx-arm64"),
            ("android", "x86_64") => Ok("onnxruntime-android-x64"),
            ("android", "aarch64") => Ok("onnxruntime-android-arm64"),
            (os, arch) => Err(format!(
                "unsupported platform for ONNX Runtime auto-download: os='{}', arch='{}'",
                os, arch
            )),
        }
    }

    fn preferred_onnxruntime_download_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(current_dir) = std::env::current_dir() {
            CoreRuntime::push_unique_path(
                &mut dirs,
                current_dir.join("voicevox_core").join("onnxruntime"),
            );
            CoreRuntime::push_unique_path(
                &mut dirs,
                current_dir
                    .join("voicevox_core")
                    .join("onnxruntime")
                    .join("lib"),
            );
            CoreRuntime::push_unique_path(&mut dirs, current_dir.join("voicevox_core"));
            CoreRuntime::push_unique_path(&mut dirs, current_dir);
        }

        if let Ok(executable_path) = std::env::current_exe()
            && let Some(executable_dir) = executable_path.parent()
        {
            let executable_dir = executable_dir.to_path_buf();
            CoreRuntime::push_unique_path(
                &mut dirs,
                executable_dir.join("voicevox_core").join("onnxruntime"),
            );
            CoreRuntime::push_unique_path(
                &mut dirs,
                executable_dir
                    .join("voicevox_core")
                    .join("onnxruntime")
                    .join("lib"),
            );
            CoreRuntime::push_unique_path(&mut dirs, executable_dir.join("voicevox_core"));
            CoreRuntime::push_unique_path(&mut dirs, executable_dir);
        }

        dirs
    }

    fn choose_onnxruntime_download_dir() -> Result<PathBuf, String> {
        let mut last_error: Option<String> = None;

        for dir in Self::preferred_onnxruntime_download_dirs() {
            match std::fs::create_dir_all(&dir) {
                Ok(_) => return Ok(dir),
                Err(e) => {
                    last_error = Some(format!("{}: {}", dir.display(), e));
                }
            }
        }

        Err(format!(
            "failed to prepare ONNX Runtime download directory: {}",
            last_error.unwrap_or_else(|| "no candidate directories available".to_string())
        ))
    }

    fn extract_onnxruntime_libraries(archive: &[u8], output_dir: &Path) -> Result<usize, String> {
        let mut tar = tar::Archive::new(GzDecoder::new(Cursor::new(archive)));
        let mut extracted = 0usize;

        for entry in tar
            .entries()
            .map_err(|e| format!("failed to read ONNX Runtime archive entries: {e}"))?
        {
            let mut entry =
                entry.map_err(|e| format!("failed to read ONNX Runtime archive entry: {e}"))?;
            let entry_path = entry
                .path()
                .map_err(|e| format!("failed to read ONNX Runtime archive entry path: {e}"))?
                .into_owned();

            let Some(file_name) = entry_path.file_name().and_then(|v| v.to_str()) else {
                continue;
            };

            if file_name != Onnxruntime::LIB_VERSIONED_FILENAME
                && file_name != Onnxruntime::LIB_UNVERSIONED_FILENAME
            {
                continue;
            }

            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).map_err(|e| {
                format!(
                    "failed to extract ONNX Runtime file '{}' from archive: {}",
                    file_name, e
                )
            })?;

            let out_path = output_dir.join(file_name);
            std::fs::write(&out_path, bytes).map_err(|e| {
                format!(
                    "failed to write ONNX Runtime file '{}': {}",
                    out_path.display(),
                    e
                )
            })?;

            extracted += 1;
        }

        Ok(extracted)
    }
}

pub(super) struct CoreRuntime {
    synthesizer: Synthesizer<OpenJtalk>,
    style_to_model: HashMap<u32, PathBuf>,
    output_sampling_rate: u32,
}

impl CoreRuntime {
    // レート制限かかるなら
    fn github_token() -> Option<String> {
        std::env::var("GH_TOKEN")
            .ok()
            .or_else(|| std::env::var("GITHUB_TOKEN").ok())
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    fn github_http_client() -> Result<&'static HttpClient, String> {
        GITHUB_HTTP_CLIENT.get_or_try_init(|| {
            HttpClient::builder()
                .timeout(Duration::from_secs(180))
                .build()
                .map_err(|e| format!("failed to build HTTP client: {e}"))
        })
    }

    async fn github_api_get<T: for<'de> Deserialize<'de>>(url: &str) -> Result<T, String> {
        let client = Self::github_http_client()?;
        let mut req = client
            .get(url)
            .header(USER_AGENT, "observer-rust/voicevox-assets")
            .header(ACCEPT, "application/vnd.github+json");

        if let Some(token) = Self::github_token() {
            req = req.header(AUTHORIZATION, format!("Bearer {}", token));
        }

        let response = req
            .send()
            .await
            .map_err(|e| format!("GitHub API request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(format!(
                "GitHub API request failed with status {} for '{}': {}",
                status,
                url,
                body.chars().take(400).collect::<String>()
            ));
        }

        response
            .json::<T>()
            .await
            .map_err(|e| format!("failed to decode GitHub API response: {e}"))
    }

    async fn fetch_release_by_tag(repo: &str, tag: &str) -> Result<GithubRelease, String> {
        let url = format!("https://api.github.com/repos/{repo}/releases/tags/{tag}");
        Self::github_api_get(&url).await
    }

    async fn fetch_releases(repo: &str) -> Result<Vec<GithubRelease>, String> {
        let url = format!("https://api.github.com/repos/{repo}/releases?per_page=100");
        Self::github_api_get(&url).await
    }

    fn parse_release_semver(tag_name: &str) -> Option<(u64, u64, u64)> {
        let normalized = tag_name.trim().trim_start_matches('v');
        let stable = normalized.split('-').next().unwrap_or(normalized);

        let mut parts = stable.split('.');
        let major = parts.next()?.parse::<u64>().ok()?;
        let minor = parts.next()?.parse::<u64>().ok()?;
        let patch = parts.next()?.parse::<u64>().ok()?;

        if parts.next().is_some() {
            return None;
        }

        Some((major, minor, patch))
    }

    async fn fetch_latest_supported_vvm_release(repo: &str) -> Result<GithubRelease, String> {
        let releases = Self::fetch_releases(repo).await?;

        let mut selected: Option<((u64, u64, u64), GithubRelease)> = None;
        for release in releases {
            if release.draft || release.prerelease {
                continue;
            }

            let Some(version) = Self::parse_release_semver(&release.tag_name) else {
                continue;
            };

            if version.0 != SUPPORTED_VVM_MAJOR || version.1 != SUPPORTED_VVM_MINOR {
                continue;
            }

            if selected
                .as_ref()
                .map(|(best, _)| version > *best)
                .unwrap_or(true)
            {
                selected = Some((version, release));
            }
        }

        selected.map(|(_, release)| release).ok_or_else(|| {
            format!(
                "no compatible VVM release found in '{}': expected '{}.{}.*'",
                repo, SUPPORTED_VVM_MAJOR, SUPPORTED_VVM_MINOR
            )
        })
    }

    fn has_extension(path: &Path, extension: &str) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
    }

    fn collect_files_with_extension(
        dir: &Path,
        extension: &str,
        dir_label: &str,
    ) -> Result<Vec<PathBuf>, String> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(dir).map_err(|e| {
            format!(
                "failed to read {} directory '{}': {e}",
                dir_label,
                dir.display()
            )
        })? {
            let entry = entry.map_err(|e| {
                format!(
                    "failed to read an entry in {} directory '{}': {e}",
                    dir_label,
                    dir.display()
                )
            })?;

            let path = entry.path();
            if Self::has_extension(&path, extension) {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    fn has_file_with_extension(
        dir: &Path,
        extension: &str,
        dir_label: &str,
    ) -> Result<bool, String> {
        if !dir.is_dir() {
            return Ok(false);
        }

        for entry in std::fs::read_dir(dir).map_err(|e| {
            format!(
                "failed to read {} directory '{}': {e}",
                dir_label,
                dir.display()
            )
        })? {
            let entry = entry.map_err(|e| {
                format!(
                    "failed to read an entry in {} directory '{}': {e}",
                    dir_label,
                    dir.display()
                )
            })?;

            if Self::has_extension(&entry.path(), extension) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn remove_existing_vvm_files(vvm_dir: &Path) -> Result<usize, String> {
        let vvm_files = Self::collect_files_with_extension(vvm_dir, "vvm", "VVM")?;

        for path in &vvm_files {
            std::fs::remove_file(path)
                .map_err(|e| format!("failed to remove old VVM file '{}': {e}", path.display()))?;
        }

        Ok(vvm_files.len())
    }

    async fn download_asset_bytes(url: &str) -> Result<Vec<u8>, String> {
        let client = Self::github_http_client()?;
        let mut req = client
            .get(url)
            .header(USER_AGENT, "observer-rust/voicevox-assets")
            .header(ACCEPT, "application/octet-stream");

        if let Some(token) = Self::github_token() {
            req = req.header(AUTHORIZATION, format!("Bearer {}", token));
        }

        let response = req
            .send()
            .await
            .map_err(|e| format!("asset download request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(format!(
                "asset download failed with status {} for '{}': {}",
                status,
                url,
                body.chars().take(400).collect::<String>()
            ));
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("failed to read downloaded asset bytes: {e}"))
    }

    async fn write_release_asset_if_exists(
        release: &GithubRelease,
        asset_name: &str,
        out_path: &Path,
        label: &str,
    ) -> Result<bool, String> {
        let Some(asset) = release.assets.iter().find(|a| a.name == asset_name) else {
            return Ok(false);
        };

        let bytes = Self::download_asset_bytes(&asset.browser_download_url).await?;
        tokio::fs::write(out_path, bytes)
            .await
            .map_err(|e| format!("failed to write {} '{}': {e}", label, out_path.display()))?;
        Ok(true)
    }

    async fn download_vvm_assets_parallel(
        release: &GithubRelease,
        vvm_dir: &Path,
    ) -> Result<usize, String> {
        let vvm_assets = release
            .assets
            .iter()
            .filter_map(|asset| {
                let lower = asset.name.to_ascii_lowercase();
                if !lower.ends_with(".vvm") {
                    return None;
                }

                Some((asset.name.clone(), asset.browser_download_url.clone()))
            })
            .collect::<Vec<_>>();

        if vvm_assets.is_empty() {
            return Err(format!(
                "no .vvm assets were found in compatible release '{}' of '{}'",
                release.tag_name, VOICEVOX_VVM_REPO
            ));
        }

        let mut join_set = tokio::task::JoinSet::new();
        let mut next_index = 0usize;
        let mut downloaded = 0usize;

        while next_index < vvm_assets.len() || !join_set.is_empty() {
            while join_set.len() < VVM_DOWNLOAD_CONCURRENCY && next_index < vvm_assets.len() {
                let (asset_name, download_url) = vvm_assets[next_index].clone();
                next_index += 1;
                let out_path = vvm_dir.join(asset_name);

                join_set.spawn(async move {
                    let bytes = Self::download_asset_bytes(&download_url).await?;
                    tokio::fs::write(&out_path, bytes).await.map_err(|e| {
                        format!("failed to write VVM file '{}': {e}", out_path.display())
                    })?;

                    Ok::<(), String>(())
                });
            }

            match join_set.join_next().await {
                Some(Ok(Ok(()))) => downloaded += 1,
                Some(Ok(Err(e))) => return Err(e),
                Some(Err(e)) => return Err(format!("failed to join VVM download task: {e}")),
                None => break,
            }
        }

        Ok(downloaded)
    }

    fn has_open_jtalk_dictionary(dict_dir: &Path) -> Result<bool, String> {
        Self::has_file_with_extension(dict_dir, "dic", "dict")
    }

    fn has_vvm_assets(vvm_dir: &Path) -> Result<bool, String> {
        Self::has_file_with_extension(vvm_dir, "vvm", "VVM")
    }

    async fn ensure_open_jtalk_dictionary(dict_dir: &Path) -> Result<(), String> {
        if Self::has_open_jtalk_dictionary(dict_dir)? {
            return Ok(());
        }

        let dict_root = dict_dir.parent().ok_or_else(|| {
            format!(
                "invalid OpenJTalk dictionary path '{}': parent directory is missing",
                dict_dir.display()
            )
        })?;

        std::fs::create_dir_all(dict_root).map_err(|e| {
            format!(
                "failed to create dictionary root directory '{}': {e}",
                dict_root.display()
            )
        })?;

        info!(
            "OpenJTalk dictionary not found at '{}'; downloading from GitHub release assets",
            dict_dir.display()
        );

        let release = Self::fetch_release_by_tag(OPEN_JTALK_REPO, OPEN_JTALK_TAG).await?;
        let asset = release
            .assets
            .iter()
            .find(|asset| asset.name == OPEN_JTALK_ASSET_NAME)
            .ok_or_else(|| {
                format!(
                    "failed to find '{}' in release '{}' of '{}'",
                    OPEN_JTALK_ASSET_NAME, release.tag_name, OPEN_JTALK_REPO
                )
            })?;

        let archive = Self::download_asset_bytes(&asset.browser_download_url).await?;
        let mut tar = tar::Archive::new(GzDecoder::new(Cursor::new(archive)));
        tar.unpack(dict_root).map_err(|e| {
            format!(
                "failed to extract OpenJTalk dictionary archive into '{}': {e}",
                dict_root.display()
            )
        })?;

        if !Self::has_open_jtalk_dictionary(dict_dir)? {
            return Err(format!(
                "OpenJTalk dictionary download finished but '{}' is still unavailable",
                dict_dir.display()
            ));
        }

        info!(
            "OpenJTalk dictionary downloaded successfully (tag={})",
            release.tag_name
        );

        Ok(())
    }

    async fn ensure_vvm_models(vvm_dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(vvm_dir).map_err(|e| {
            format!(
                "failed to create VVM directory '{}': {e}",
                vvm_dir.display()
            )
        })?;

        let models_root = vvm_dir.parent().unwrap_or(vvm_dir);
        let marker_path = models_root.join(VVM_RELEASE_MARKER_FILE);
        let has_local_assets = Self::has_vvm_assets(vvm_dir)?;

        let release = match Self::fetch_latest_supported_vvm_release(VOICEVOX_VVM_REPO).await {
            Ok(release) => release,
            Err(e) => {
                if has_local_assets {
                    warn!(
                        "failed to fetch compatible VVM release metadata: {}. using existing local VVMs under '{}'",
                        e,
                        vvm_dir.display()
                    );
                    return Ok(());
                }
                return Err(e);
            }
        };

        let current_tag = std::fs::read_to_string(&marker_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if has_local_assets && current_tag.as_deref() == Some(release.tag_name.as_str()) {
            return Ok(());
        }

        if has_local_assets {
            info!(
                "refreshing VVM assets in '{}' to compatible release '{}' (previous marker={})",
                vvm_dir.display(),
                release.tag_name,
                current_tag.unwrap_or_else(|| "<none>".to_string())
            );

            let removed = Self::remove_existing_vvm_files(vvm_dir)?;
            if removed > 0 {
                info!("removed {} old VVM file(s) before refresh", removed);
            }
        } else {
            info!(
                "VVM assets not found at '{}'; downloading compatible release '{}'",
                vvm_dir.display(),
                release.tag_name
            );
        }

        let downloaded = Self::download_vvm_assets_parallel(&release, vvm_dir).await?;

        let readme_path = models_root.join(MODELS_README_FILE);
        Self::write_release_asset_if_exists(
            &release,
            MODELS_README_FILE,
            &readme_path,
            "models README",
        )
        .await?;

        let terms_path = models_root.join(MODELS_TERMS_FILE);
        Self::write_release_asset_if_exists(
            &release,
            MODELS_TERMS_FILE,
            &terms_path,
            "models terms",
        )
        .await?;

        if !Self::has_vvm_assets(vvm_dir)? {
            return Err(format!(
                "VVM download finished but '{}' is still unavailable",
                vvm_dir.display()
            ));
        }

        info!(
            "VVM assets downloaded successfully (tag={}, files={})",
            release.tag_name, downloaded
        );

        std::fs::write(&marker_path, format!("{}\n", release.tag_name)).map_err(|e| {
            format!(
                "failed to write VVM release marker '{}': {e}",
                marker_path.display()
            )
        })?;

        Ok(())
    }

    async fn ensure_required_assets(config: &VoiceCoreConfig) -> Result<(), String> {
        VoiceVoxInitDownloader::prepare(config).await
    }

    fn parse_acceleration_mode(mode: &str) -> AccelerationMode {
        match mode.trim().to_ascii_lowercase().as_str() {
            "auto" => {
                info!(
                    "VOICEVOX acceleration 'auto' is mapped to 'cpu' to avoid ORT mixed-provider warnings. Set VOICEVOX_CORE_ACCELERATION=gpu to force GPU."
                );
                AccelerationMode::Cpu
            }
            "cpu" => AccelerationMode::Cpu,
            "gpu" => AccelerationMode::Gpu,
            other => {
                warn!(
                    "unknown VOICEVOX acceleration mode '{}', fallback to auto",
                    other
                );
                AccelerationMode::Auto
            }
        }
    }

    fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
        if !paths.contains(&path) {
            paths.push(path);
        }
    }

    fn push_unique_candidate(candidates: &mut Vec<String>, candidate: String) {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    fn onnxruntime_search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(current_dir) = std::env::current_dir() {
            Self::push_unique_path(&mut dirs, current_dir.clone());
            Self::push_unique_path(&mut dirs, current_dir.join("voicevox_core"));
            Self::push_unique_path(
                &mut dirs,
                current_dir.join("voicevox_core").join("onnxruntime"),
            );
            Self::push_unique_path(
                &mut dirs,
                current_dir
                    .join("voicevox_core")
                    .join("onnxruntime")
                    .join("lib"),
            );
        }

        if let Ok(executable_path) = std::env::current_exe()
            && let Some(executable_dir) = executable_path.parent()
        {
            let executable_dir = executable_dir.to_path_buf();
            Self::push_unique_path(&mut dirs, executable_dir.clone());
            Self::push_unique_path(&mut dirs, executable_dir.join("voicevox_core"));
            Self::push_unique_path(
                &mut dirs,
                executable_dir.join("voicevox_core").join("onnxruntime"),
            );
            Self::push_unique_path(
                &mut dirs,
                executable_dir
                    .join("voicevox_core")
                    .join("onnxruntime")
                    .join("lib"),
            );
        }

        dirs
    }

    fn build_auto_onnxruntime_candidates() -> Vec<String> {
        let mut candidates = Vec::new();

        for dir in Self::onnxruntime_search_dirs() {
            for filename in [
                Onnxruntime::LIB_VERSIONED_FILENAME,
                Onnxruntime::LIB_UNVERSIONED_FILENAME,
            ] {
                let path = dir.join(filename);
                if path.is_file() {
                    Self::push_unique_candidate(
                        &mut candidates,
                        path.to_string_lossy().into_owned(),
                    );
                }
            }
        }

        Self::push_unique_candidate(
            &mut candidates,
            Onnxruntime::LIB_VERSIONED_FILENAME.to_string(),
        );
        Self::push_unique_candidate(
            &mut candidates,
            Onnxruntime::LIB_UNVERSIONED_FILENAME.to_string(),
        );
        candidates
    }

    fn build_configured_onnxruntime_candidates(configured: &str) -> Vec<String> {
        let configured = configured.trim();
        if configured.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let configured_path = PathBuf::from(configured);

        if configured_path.is_file() {
            Self::push_unique_candidate(
                &mut candidates,
                configured_path.to_string_lossy().into_owned(),
            );
        }

        if !configured_path.is_absolute() {
            if let Ok(current_dir) = std::env::current_dir() {
                let from_current_dir = current_dir.join(configured);
                if from_current_dir.is_file() {
                    Self::push_unique_candidate(
                        &mut candidates,
                        from_current_dir.to_string_lossy().into_owned(),
                    );
                }
            }

            if let Ok(executable_path) = std::env::current_exe()
                && let Some(executable_dir) = executable_path.parent()
            {
                let from_executable_dir = executable_dir.join(configured);
                if from_executable_dir.is_file() {
                    Self::push_unique_candidate(
                        &mut candidates,
                        from_executable_dir.to_string_lossy().into_owned(),
                    );
                }

                let from_voicevox_core = executable_dir.join("voicevox_core").join(configured);
                if from_voicevox_core.is_file() {
                    Self::push_unique_candidate(
                        &mut candidates,
                        from_voicevox_core.to_string_lossy().into_owned(),
                    );
                }
            }
        }

        // 最後に元の値を足して、PATH解決やモジュール名解決も試せるようにする。
        Self::push_unique_candidate(&mut candidates, configured.to_string());
        candidates
    }

    async fn try_load_onnxruntime_with_candidates(
        candidates: Vec<String>,
    ) -> Result<(&'static Onnxruntime, String), Vec<String>> {
        let mut failed_attempts = Vec::new();

        for candidate in candidates {
            match Onnxruntime::load_once()
                .filename(candidate.clone())
                .perform()
                .await
            {
                Ok(ort) => return Ok((ort, candidate)),
                Err(e) => failed_attempts.push(format!("{} => {}", candidate, e)),
            }
        }

        Err(failed_attempts)
    }

    async fn load_onnxruntime(config: &VoiceCoreConfig) -> Result<&'static Onnxruntime, String> {
        if let Some(existing) = Onnxruntime::get() {
            return Ok(existing);
        }

        let configured = config.onnxruntime_filename.trim();
        if !configured.is_empty() {
            let configured_candidates = Self::build_configured_onnxruntime_candidates(configured);
            match Self::try_load_onnxruntime_with_candidates(configured_candidates).await {
                Ok((ort, loaded_from)) => {
                    info!(
                        "ONNX Runtime loaded from VOICEVOX_ONNXRUNTIME_FILENAME candidate: {}",
                        loaded_from
                    );
                    return Ok(ort);
                }
                Err(failed_attempts) => {
                    return Err(format!(
                        "failed to load ONNX Runtime using VOICEVOX_ONNXRUNTIME_FILENAME='{}' (os='{}', arch='{}', attempts=[{}])",
                        configured,
                        std::env::consts::OS,
                        std::env::consts::ARCH,
                        failed_attempts.join(" | ")
                    ));
                }
            }
        }

        let auto_candidates = Self::build_auto_onnxruntime_candidates();
        if !auto_candidates.is_empty() {
            match Self::try_load_onnxruntime_with_candidates(auto_candidates).await {
                Ok((ort, loaded_from)) => {
                    info!(
                        "ONNX Runtime auto-loaded with runtime linking from '{}'",
                        loaded_from
                    );
                    return Ok(ort);
                }
                Err(failed_attempts) => {
                    warn!(
                        "automatic ONNX Runtime load attempts failed; fallback to default loader path (attempts=[{}])",
                        failed_attempts.join(" | ")
                    );
                }
            }
        }

        Onnxruntime::load_once().perform().await.map_err(|e| {
            format!(
                "failed to auto-load ONNX Runtime in runtime-link mode: {} (os='{}', arch='{}', default='{}')",
                e,
                std::env::consts::OS,
                std::env::consts::ARCH,
                Onnxruntime::LIB_VERSIONED_FILENAME
            )
        })
    }

    fn discover_vvm_files(vvm_dir: &Path) -> Result<Vec<PathBuf>, String> {
        if !vvm_dir.exists() {
            return Err(format!(
                "VVM directory does not exist: {}",
                vvm_dir.display()
            ));
        }
        if !vvm_dir.is_dir() {
            return Err(format!(
                "VVM path is not a directory: {}",
                vvm_dir.display()
            ));
        }

        let files = Self::collect_files_with_extension(vvm_dir, "vvm", "VVM")?;

        if files.is_empty() {
            return Err(format!("no .vvm files found under '{}'", vvm_dir.display()));
        }

        Ok(files)
    }

    fn collect_style_ids(metas: &[CharacterMeta]) -> Vec<u32> {
        let mut ids = metas
            .iter()
            .flat_map(|ch| ch.styles.iter().map(|style| style.id.0))
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn collect_version_key(metas: &[CharacterMeta]) -> String {
        let mut versions = metas
            .iter()
            .map(|ch| ch.version.0.clone())
            .collect::<Vec<_>>();
        versions.sort();
        versions.dedup();
        versions.join("|")
    }

    fn build_speaker_not_found_error(&self, speaker: u32) -> String {
        let mut styles = self.style_to_model.keys().copied().collect::<Vec<_>>();
        styles.sort_unstable();

        let preview_limit = 20usize;
        let preview = styles
            .iter()
            .take(preview_limit)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        if styles.is_empty() {
            return format!(
                "speaker(style id) {} is not available: no styles were indexed from VVM files",
                speaker
            );
        }

        if styles.len() > preview_limit {
            return format!(
                "speaker(style id) {} is not available. available styles (first {} of {}): {}",
                speaker,
                preview_limit,
                styles.len(),
                preview
            );
        }

        format!(
            "speaker(style id) {} is not available. available styles: {}",
            speaker, preview
        )
    }

    pub(super) async fn new(config: &VoiceCoreConfig) -> Result<Self, String> {
        Self::ensure_required_assets(config).await?;

        let ort = Self::load_onnxruntime(config).await?;
        let acceleration = Self::parse_acceleration_mode(&config.acceleration_mode);

        let open_jtalk = OpenJtalk::new(&config.open_jtalk_dict_dir)
            .await
            .map_err(|e| {
                format!(
                    "failed to initialize OpenJTalk dictionary '{}': {}",
                    config.open_jtalk_dict_dir, e
                )
            })?;

        let synthesizer = Synthesizer::builder(ort)
            .text_analyzer(open_jtalk)
            .acceleration_mode(acceleration)
            .cpu_num_threads(config.cpu_threads)
            .build()
            .map_err(|e| format!("voicevox synthesizer build failed: {e}"))?;

        let vvm_dir = PathBuf::from(&config.vvm_dir);
        let vvm_files = Self::discover_vvm_files(&vvm_dir)?;

        let mut discovered = Vec::<DiscoveredModel>::new();
        let mut version_counts = HashMap::<String, usize>::new();
        let mut metadata_open_failed = 0usize;

        for model_path in vvm_files {
            let model = match VoiceModelFile::open(&model_path).await {
                Ok(model) => model,
                Err(e) => {
                    metadata_open_failed += 1;
                    warn!(
                        "failed to open VVM '{}' for metadata: {}; skipping",
                        model_path.display(),
                        e
                    );
                    continue;
                }
            };

            let style_ids = Self::collect_style_ids(model.metas());
            if style_ids.is_empty() {
                warn!(
                    "VVM '{}' has no styles in metadata; skipping",
                    model_path.display()
                );
                continue;
            }

            let version_key = Self::collect_version_key(model.metas());
            *version_counts.entry(version_key.clone()).or_insert(0) += 1;

            discovered.push(DiscoveredModel {
                model_path,
                voice_model: model,
                style_ids,
                version_key,
            });
        }

        if discovered.is_empty() {
            return Err(format!(
                "no usable style IDs were found in VVM files under '{}' (metadata_open_failed={})",
                config.vvm_dir, metadata_open_failed
            ));
        }

        if metadata_open_failed > 0 {
            warn!(
                "skipped {} VVM file(s) because metadata could not be read",
                metadata_open_failed
            );
        }

        let selected_version = version_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(version, _)| version.clone())
            .ok_or_else(|| "failed to determine model version set".to_string())?;

        if version_counts.len() > 1 {
            let details = version_counts
                .iter()
                .map(|(k, v)| format!("{}: {} files", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            warn!(
                "detected mixed VVM character versions ({}). Using '{}' only to avoid voicevox_core meta mismatch warnings.",
                details, selected_version
            );
        }

        let discovered_total = discovered.len();
        let selected_models = discovered
            .into_iter()
            .filter(|model| model.version_key == selected_version)
            .collect::<Vec<_>>();
        let skipped_models = discovered_total.saturating_sub(selected_models.len());

        if selected_models.is_empty() {
            return Err(format!(
                "no usable style IDs remained after version filtering under '{}'",
                config.vvm_dir
            ));
        }

        if skipped_models > 0 {
            warn!(
                "skipped {} VVM file(s) due to version mismatch with selected '{}'",
                skipped_models, selected_version
            );
        }

        if !config.load_all_models {
            warn!(
                "VOICEVOX load_all_models=false is ignored in concurrent mode; preloading all selected VVMs"
            );
        }

        let mut style_to_model = HashMap::new();
        let mut loaded_models = 0usize;
        let mut load_failed_models = 0usize;

        for discovered_model in selected_models {
            let model_path = discovered_model.model_path.clone();

            if let Err(e) = synthesizer
                .load_voice_model(&discovered_model.voice_model)
                .perform()
                .await
            {
                load_failed_models += 1;
                warn!(
                    "failed to load VVM '{}': {}; skipping",
                    model_path.display(),
                    e
                );
                continue;
            }

            loaded_models += 1;

            for style_id in &discovered_model.style_ids {
                if let Some(prev) = style_to_model.insert(*style_id, model_path.clone())
                    && prev != model_path
                {
                    warn!(
                        "style id {} appears in multiple VVMs: '{}' and '{}'; using latest",
                        style_id,
                        prev.display(),
                        model_path.display()
                    );
                }
            }
        }

        if style_to_model.is_empty() {
            return Err(format!(
                "failed to load any usable VVM under '{}' (metadata_open_failed={}, load_failed={})",
                config.vvm_dir, metadata_open_failed, load_failed_models
            ));
        }

        if load_failed_models > 0 {
            warn!(
                "skipped {} VVM file(s) because model loading failed",
                load_failed_models
            );
        }

        info!(
            "VOICEVOX CORE initialized (acceleration={}, cpu_threads={}, load_all_models=forced(true), output_sampling_rate={}, output_stereo=fixed(false), dict_dir={}, vvm_dir={}, model_count={}, style_count={})",
            if config.acceleration_mode.trim().eq_ignore_ascii_case("auto") {
                "cpu(auto-mapped)".to_string()
            } else {
                config.acceleration_mode.clone()
            },
            config.cpu_threads,
            config.output_sampling_rate,
            config.open_jtalk_dict_dir,
            config.vvm_dir,
            loaded_models,
            style_to_model.len()
        );

        Ok(Self {
            synthesizer,
            style_to_model,
            output_sampling_rate: config.output_sampling_rate,
        })
    }

    pub(super) async fn synthesize(
        &self,
        text: &str,
        speaker: u32,
        speed_scale: Option<f32>,
        pitch_scale: Option<f32>,
    ) -> Result<Vec<u8>, String> {
        if !self.style_to_model.contains_key(&speaker) {
            return Err(self.build_speaker_not_found_error(speaker));
        }

        let style_id = StyleId::new(speaker);
        let mut query: AudioQuery = self
            .synthesizer
            .create_audio_query(text, style_id)
            .await
            .map_err(|e| format!("failed to create audio query: {e}"))?;

        // VOICEVOXの推奨サンプリングレートを尊重する（警告抑止）。
        // Discord向け品質を優先して出力条件を固定する。
        query.output_sampling_rate = self.output_sampling_rate;
        query.output_stereo = false;
        query.post_phoneme_length = 0.05;
        query.pre_phoneme_length = 0.05;

        if let Some(speed_scale) = speed_scale {
            query.speed_scale = speed_scale;
        }

        if let Some(pitch_scale) = pitch_scale {
            query.pitch_scale = pitch_scale;
        }

        self.synthesizer
            .synthesis(&query, style_id)
            .enable_interrogative_upspeak(true)
            .perform()
            .await
            .map_err(|e| format!("voice synthesis failed: {e}"))
    }
}
