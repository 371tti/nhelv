use std::borrow::Cow;

use chrono::{FixedOffset, TimeZone, Utc};
use kanalizer::{ConvertOptions, Kanalizer};
use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use serenity::{
    all::MessageFlags,
    cache::Cache,
    model::channel::Message,
    utils::{ContentSafeOptions, content_safe},
};

static RE_MASKED_LINK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[([^\]]+)\]\((https?://[^)\s]+)\)").unwrap());

static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"https?://\S+").unwrap());

static RE_CODE_BLOCK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```.*?```").unwrap());

static RE_INLINE_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"`[^`\n]+`").unwrap());

static RE_SPOILER: Lazy<Regex> = Lazy::new(|| Regex::new(r"\|\|(.+?)\|\|").unwrap());

static RE_SLASH_COMMAND: Lazy<Regex> = Lazy::new(|| Regex::new(r"</([^:>]+):\d+>").unwrap());

static RE_CUSTOM_EMOJI: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<a?:([A-Za-z0-9_]+):\d+>").unwrap());

static RE_TIMESTAMP: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<t:(\d+)(?::([tTdDfFsSR]))?>").unwrap());

static RE_GUILD_NAV: Lazy<Regex> = Lazy::new(|| Regex::new(r"<id:[^>]+>").unwrap());

static RE_HEADER: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s{0,3}#{1,3}\s+").unwrap());

static RE_SUBTEXT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*-#(?:\s+.*)?$").unwrap());

static RE_BLOCKQUOTE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*>>?>\s?").unwrap());

static RE_LIST: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*(?:[-*]|\d+\.)\s+").unwrap());

const TTS_SEGMENT_SOFT_LIMIT: usize = 36;
const TTS_SEGMENT_HARD_LIMIT: usize = 60;
const TTS_MAX_TOTAL_CHARS: usize = 200;

thread_local! {
    static EN2KANA_DECODER: EN2KANA = EN2KANA::new();
}

pub struct EN2KANA {
    decoder: Kanalizer,
}

impl EN2KANA {
    pub fn new() -> Self {
        Self {
            decoder: Kanalizer::new(),
        }
    }

    fn normalize_ascii_letters(letters: &str) -> String {
        let mut normalized = String::with_capacity(letters.len() + 4);
        let mut prev_was_lower = false;

        for ch in letters.bytes() {
            let is_upper = ch.is_ascii_uppercase();
            if prev_was_lower && is_upper {
                normalized.push(' ');
            }

            normalized.push(char::from(ch.to_ascii_lowercase()));
            prev_was_lower = ch.is_ascii_lowercase();
        }

        normalized
    }

    fn digit_to_kana(digit: u8) -> &'static str {
        match digit {
            b'0' => "ゼロ",
            b'1' => "ワン",
            b'2' => "ツー",
            b'3' => "スリー",
            b'4' => "フォー",
            b'5' => "ファイブ",
            b'6' => "シックス",
            b'7' => "セブン",
            b'8' => "エイト",
            b'9' => "ナイン",
            _ => "",
        }
    }

    fn should_skip_kanalyze_ascii_word(word: &str) -> bool {
        !word.is_empty() && word.len() <= 5 && word.bytes().all(|b| b.is_ascii_uppercase())
    }

    fn convert_ascii_alnum_run(&self, run: &str, opt: &ConvertOptions) -> Result<String, String> {
        let bytes = run.as_bytes();
        let has_alpha = bytes.iter().any(|b| b.is_ascii_alphabetic());
        if !has_alpha {
            return Ok(run.to_string());
        }

        let mut result = String::with_capacity(run.len() + 8);
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i].is_ascii_alphabetic() {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }

                let alpha_word = &run[start..i];
                if Self::should_skip_kanalyze_ascii_word(alpha_word) {
                    result.push_str(alpha_word);
                    continue;
                }

                let normalized = Self::normalize_ascii_letters(alpha_word);
                let converted = self
                    .decoder
                    .convert(&normalized, opt)
                    .map_err(|e| e.to_string())?;
                result.push_str(&converted);
                continue;
            }

            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }

            for digit in run[start..i].bytes() {
                result.push_str(Self::digit_to_kana(digit));
            }
        }

        Ok(result)
    }

    pub fn convert(&self, src: &str, opt: Option<ConvertOptions>) -> Result<String, String> {
        if src.is_empty() {
            return Ok(String::new());
        }

        if !src.as_bytes().iter().any(|b| b.is_ascii_alphabetic()) {
            return Ok(src.to_string());
        }

        let opt = opt.unwrap_or_default();
        let mut result = String::with_capacity(src.len());
        let bytes = src.as_bytes();
        let mut cursor = 0;
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i].is_ascii_alphanumeric() {
                if cursor < i {
                    result.push_str(&src[cursor..i]);
                }

                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                    i += 1;
                }

                let converted = self.convert_ascii_alnum_run(&src[start..i], &opt)?;
                result.push_str(&converted);
                cursor = i;
                continue;
            }

            i += 1;
        }

        if cursor < src.len() {
            result.push_str(&src[cursor..]);
        }

        Ok(result)
    }
}

/// 1番初めに適用
/// 読み上げるべきかDiscordMsgのFlagsや内容を見て判断する
pub fn build_tts_text_from_message(cache: &Cache, msg: &Message) -> Option<String> {
    // 先頭 ! / / はコマンド扱いで無視
    if msg.content.starts_with('!') || msg.content.starts_with('/') {
        return None;
    }

    if msg
        .flags
        .map(|flags| flags.contains(MessageFlags::SUPPRESS_NOTIFICATIONS))
        .unwrap_or(false)
    {
        return None;
    }

    // serenity 側で mention 系をまず人間向け文字列へ
    let safe = if let Some(guild_id) = msg.guild_id {
        let options = ContentSafeOptions::default().display_as_member_from(guild_id);
        content_safe(cache, &msg.content, &options, &msg.mentions)
    } else {
        msg.content_safe(cache)
    };

    let content = normalize_tts_text(&safe);

    let content = if content.is_empty() {
        if msg.attachments.is_empty() {
            return None;
        }
        "ファイルが送信されました".to_string()
    } else {
        content
    };

    Some(content)
}

/// 2番目に適用
/// 辞書適用
pub fn apply_tts_dictionary(input: &str, entries: &[(String, String)]) -> String {
    if input.is_empty() || entries.is_empty() {
        return input.to_string();
    }

    // 長いキーを先に適用して、短いキーによる過剰置換を減らす。
    let mut sorted = entries
        .iter()
        .filter(|(source, target)| !source.is_empty() && !target.is_empty())
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()));

    let mut out = input.to_string();
    for (source, target) in sorted {
        out = out.replace(source, target);
    }

    out
}

/// 3番目に適用
/// Discord のフォーマットを潰す
pub fn normalize_tts_text(input: &str) -> String {
    // Discord の見た目に近い順で潰す
    // code block を spoiler より先に処理すると挙動が自然
    let mut out = input.to_string();

    // [text](url) -> text
    if out.contains("](") {
        out = regex_replace_all(out, &RE_MASKED_LINK, "$1");
    }

    // ```...``` -> コードブロック
    if out.contains("```") {
        out = regex_replace_all(out, &RE_CODE_BLOCK, "コードブロック");
    }

    // `...` -> コード
    if out.contains('`') {
        out = regex_replace_all(out, &RE_INLINE_CODE, "コード");
    }

    // ||...|| -> ネタバレあり
    if out.contains("||") {
        out = regex_replace_all_with(out, &RE_SPOILER, |_caps: &Captures| {
            "スポイラー".to_string()
        });
    }

    // </foo:123> / </foo bar:123> -> スラッシュコマンド foo / foo bar
    if out.contains("</") {
        out = regex_replace_all_with(out, &RE_SLASH_COMMAND, |caps: &Captures| {
            let name = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            if name.is_empty() {
                "スラッシュコマンド".to_string()
            } else {
                format!("スラッシュコマンド {}", name)
            }
        });
    }

    // <:name:id> / <a:name:id> -> 絵文字 name
    if out.contains("<:") || out.contains("<a:") {
        out = regex_replace_all_with(out, &RE_CUSTOM_EMOJI, |caps: &Captures| {
            let name = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            if name.is_empty() {
                "絵文字".to_string()
            } else {
                format!("絵文字 {}", name)
            }
        });
    }

    // <t:1618953630:R> -> 相対時刻 / 日時
    if out.contains("<t:") {
        out = regex_replace_all_with(out, &RE_TIMESTAMP, |caps: &Captures| {
            let ts = caps.get(1).and_then(|m| m.as_str().parse::<i64>().ok());

            let style = caps.get(2).map(|m| m.as_str());

            match ts {
                Some(ts) => format_timestamp_for_tts(ts, style),
                None => "日時".to_string(),
            }
        });
    }

    // <id:guide> など
    if out.contains("<id:") {
        out = regex_replace_all(out, &RE_GUILD_NAV, "サーバー内リンク");
    }

    // URL
    if out.contains("http://") || out.contains("https://") {
        out = regex_replace_all(out, &RE_URL, "URL");
    }

    // Markdown 記号のうち読み上げ不要なもの
    if out.contains('#') {
        out = regex_replace_all(out, &RE_HEADER, "");
    }
    if out.contains("-#") {
        out = regex_replace_all(out, &RE_SUBTEXT, "");
    }
    if out.contains('>') {
        out = regex_replace_all(out, &RE_BLOCKQUOTE, "");
    }
    if out.contains('\n') || out.starts_with("- ") || out.starts_with("* ") {
        out = regex_replace_all(out, &RE_LIST, "");
    }

    // 改行・連続空白を畳む
    out = collapse_whitespace_for_tts(&out);

    // <数字>.<数字> / .<数字> は「テン」読みとして扱い、
    // 小数点以下は桁読み（例: .32 -> テンサンニイ）に固定する。
    out = normalize_decimal_points_for_tts(&out);

    // OpenJTalk/VOICEVOX で句頭に置くと壊れやすい記号を落とす
    out = strip_unsafe_head_symbols_for_tts(&out);

    // 最後に英数字連続区間をカナ寄せして読み上げを安定化する
    out = normalize_ascii_alnum_to_kana(&out);

    if out.chars().count() > TTS_MAX_TOTAL_CHARS {
        let truncated = out.chars().take(TTS_MAX_TOTAL_CHARS).collect::<String>();
        return format!("{}、以下省略", truncated);
    }

    out
}

/// 4番目に適用
/// TTS の発話内容として適切な長さで分割する
pub fn split_tts_segments(input: &str) -> Vec<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = String::with_capacity(TTS_SEGMENT_HARD_LIMIT + 8);
    let mut current_len = 0usize;

    for ch in trimmed.chars() {
        if matches!(ch, '\n' | '\r') {
            let chunk = current.trim();
            if !chunk.is_empty() {
                segments.push(chunk.to_string());
            }
            current.clear();
            current_len = 0;
            continue;
        }

        current.push(ch);
        current_len += 1;

        let at_sentence_boundary = matches!(
            ch,
            '。' | '！'
                | '？'
                | '!'
                | '?'
                | '、'
                | ','
                | '，'
                | '．'
                | '.'
                | '；'
                | ';'
                | '：'
                | ':'
        );
        let at_soft_boundary = matches!(ch, ' ' | '　') && current_len >= TTS_SEGMENT_SOFT_LIMIT;
        let over_hard_limit = current_len >= TTS_SEGMENT_HARD_LIMIT;

        if at_sentence_boundary || at_soft_boundary || over_hard_limit {
            let chunk = current.trim();
            if !chunk.is_empty() {
                segments.push(chunk.to_string());
            }
            current.clear();
            current_len = 0;
        }
    }

    let rest = current.trim();
    if !rest.is_empty() {
        segments.push(rest.to_string());
    }

    segments
}

fn format_timestamp_for_tts(ts: i64, style: Option<&str>) -> String {
    // Discord 自体は各ユーザーの timezone / locale で表示するが、
    // 読み上げは固定表現の方が扱いやすいので JST に寄せる
    let jst = FixedOffset::east_opt(9 * 60 * 60).unwrap();

    let Some(dt_utc) = Utc.timestamp_opt(ts, 0).single() else {
        return "日時".to_string();
    };
    let dt = dt_utc.with_timezone(&jst);

    match style.unwrap_or("f") {
        "t" => dt.format("%H時%M分").to_string(),
        "T" => dt.format("%H時%M分%S秒").to_string(),
        "d" => dt.format("%Y年%m月%d日").to_string(),
        "D" => dt.format("%Y年%m月%d日").to_string(),
        "f" => dt.format("%Y年%m月%d日 %H時%M分").to_string(),
        "F" => dt.format("%Y年%m月%d日 %H時%M分").to_string(),
        "s" => dt.format("%Y年%m月%d日 %H時%M分").to_string(),
        "S" => dt.format("%Y年%m月%d日 %H時%M分%S秒").to_string(),
        "R" => relative_time_jp(ts),
        _ => dt.format("%Y年%m月%d日 %H時%M分").to_string(),
    }
}

fn relative_time_jp(ts: i64) -> String {
    let now = Utc::now().timestamp();
    let diff = ts - now;
    let abs = diff.abs();

    let (value, unit) = if abs < 60 {
        (abs, "秒")
    } else if abs < 60 * 60 {
        (abs / 60, "分")
    } else if abs < 60 * 60 * 24 {
        (abs / (60 * 60), "時間")
    } else if abs < 60 * 60 * 24 * 30 {
        (abs / (60 * 60 * 24), "日")
    } else if abs < 60 * 60 * 24 * 365 {
        (abs / (60 * 60 * 24 * 30), "か月")
    } else {
        (abs / (60 * 60 * 24 * 365), "年")
    };

    if diff >= 0 {
        format!("{}{}後", value, unit)
    } else {
        format!("{}{}前", value, unit)
    }
}

fn is_tts_phrase_boundary(c: char) -> bool {
    matches!(
        c,
        ' ' | '　'
            | '\n'
            | '\r'
            | '\t'
            | '、'
            | '。'
            | ','
            | '，'
            | '.'
            | '．'
            | ';'
            | '；'
            | ':'
            | '：'
            | '!'
            | '！'
            | '?'
            | '？'
    )
}

fn is_tts_head_unsafe_symbol(c: char) -> bool {
    matches!(
        c,
        '?' | '？'
            | '!'
            | '！'
            | '_'
            | 'ー'
            | '－'
            | '—'
            | '―'
            | '…'
            | '‥'
            | '、'
            | '。'
            | ','
            | '，'
            | '.'
            | '．'
            | ';'
            | '；'
            | ':'
            | '：'
    )
}

fn is_tts_seed_char(c: char) -> bool {
    !c.is_whitespace()
        && !matches!(
            c,
            '?' | '？'
                | '!'
                | '！'
                | '_'
                | 'ー'
                | '－'
                | '—'
                | '―'
                | '…'
                | '‥'
                | '、'
                | '。'
                | ','
                | '，'
                | '.'
                | '．'
                | ';'
                | '；'
                | ':'
                | '：'
                | '['
                | ']'
                | '('
                | ')'
                | '{'
                | '}'
                | '<'
                | '>'
                | '"'
                | '\''
                | '`'
                | '|'
        )
}

fn strip_unsafe_head_symbols_for_tts(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut has_seed_in_phrase = false;

    for c in input.chars() {
        if c.is_whitespace() {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            has_seed_in_phrase = false;
            continue;
        }

        if is_tts_head_unsafe_symbol(c) && !has_seed_in_phrase {
            continue;
        }

        out.push(c);

        if is_tts_phrase_boundary(c) {
            has_seed_in_phrase = false;
        } else if is_tts_seed_char(c) {
            has_seed_in_phrase = true;
        }
    }

    let trimmed_len = out.trim_end().len();
    if trimmed_len != out.len() {
        out.truncate(trimmed_len);
    }

    out
}

fn normalize_ascii_alnum_to_kana(input: &str) -> String {
    if !input.as_bytes().iter().any(|b| b.is_ascii_alphabetic()) {
        return input.to_string();
    }

    EN2KANA_DECODER.with(|decoder| {
        decoder
            .convert(input, None)
            .unwrap_or_else(|_| input.to_string())
    })
}

fn normalize_decimal_points_for_tts(input: &str) -> String {
    if !input.as_bytes().contains(&b'.') {
        return input.to_string();
    }

    let mut out = String::with_capacity(input.len() + 8);
    let mut chars = input.chars().peekable();
    let mut prev_char: Option<char> = None;
    let mut changed = false;

    while let Some(ch) = chars.next() {
        if ch == '.' {
            let next_is_digit = chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false);
            let left_is_decimal_ok = match prev_char {
                None => true,
                Some(prev) => prev.is_ascii_digit() || !prev.is_ascii_alphanumeric(),
            };

            if next_is_digit && left_is_decimal_ok {
                out.push_str("テン");
                changed = true;

                let mut consumed = false;
                while let Some(next) = chars.peek().copied() {
                    if !next.is_ascii_digit() {
                        break;
                    }

                    out.push_str(digit_to_decimal_kana(next));
                    chars.next();
                    consumed = true;
                }

                if consumed {
                    prev_char = Some('0');
                }
                continue;
            }
        }

        out.push(ch);
        prev_char = Some(ch);
    }

    if changed { out } else { input.to_string() }
}

fn digit_to_decimal_kana(digit: char) -> &'static str {
    match digit {
        '0' => "ゼロ",
        '1' => "イチ",
        '2' => "ニイ",
        '3' => "サン",
        '4' => "ヨン",
        '5' => "ゴ",
        '6' => "ロク",
        '7' => "ナナ",
        '8' => "ハチ",
        '9' => "キュウ",
        _ => "",
    }
}

fn collapse_whitespace_for_tts(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_was_space = true;

    for ch in input.chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                out.push(' ');
                prev_was_space = true;
            }
            continue;
        }

        out.push(ch);
        prev_was_space = false;
    }

    if out.ends_with(' ') {
        out.pop();
    }

    out
}

fn regex_replace_all(text: String, re: &Regex, replacement: &str) -> String {
    match re.replace_all(&text, replacement) {
        Cow::Borrowed(_) => text,
        Cow::Owned(owned) => owned,
    }
}

fn regex_replace_all_with(
    text: String,
    re: &Regex,
    mut replacer: impl FnMut(&Captures) -> String,
) -> String {
    match re.replace_all(&text, |caps: &Captures| replacer(caps)) {
        Cow::Borrowed(_) => text,
        Cow::Owned(owned) => owned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_uppercase_ascii_word_is_skipped_for_kanalyze() {
        assert!(EN2KANA::should_skip_kanalyze_ascii_word("CPU"));
        assert!(EN2KANA::should_skip_kanalyze_ascii_word("ABCDE"));
        assert!(!EN2KANA::should_skip_kanalyze_ascii_word("ABCDEF"));
        assert!(!EN2KANA::should_skip_kanalyze_ascii_word("Gpu"));
    }

    #[test]
    fn convert_keeps_short_uppercase_ascii_word() {
        let conv = EN2KANA::new();
        let out = conv
            .convert("CPU usage", None)
            .expect("EN2KANA conversion should succeed");

        assert!(out.contains("CPU"));
    }

    #[test]
    fn decimal_fraction_is_read_digit_by_digit() {
        assert_eq!(normalize_tts_text(".32"), "テンサンニイ");
        assert_eq!(normalize_tts_text("1.32"), "1テンサンニイ");
    }
}
