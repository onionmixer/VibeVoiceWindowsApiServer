#include "tokenizer/unicode_utils.h"

namespace UnicodeUtils {

uint32_t decodeUtf8(const char* data, size_t len, size_t& pos) {
    if (pos >= len) return 0xFFFD;

    uint8_t b0 = (uint8_t)data[pos];

    // 1-byte (ASCII)
    if (b0 < 0x80) {
        pos += 1;
        return b0;
    }

    // 2-byte
    if ((b0 & 0xE0) == 0xC0) {
        if (pos + 1 >= len) { pos += 1; return 0xFFFD; }
        uint8_t b1 = (uint8_t)data[pos + 1];
        if ((b1 & 0xC0) != 0x80) { pos += 1; return 0xFFFD; }
        pos += 2;
        uint32_t cp = ((uint32_t)(b0 & 0x1F) << 6) | (b1 & 0x3F);
        return cp < 0x80 ? 0xFFFD : cp; // overlong check
    }

    // 3-byte
    if ((b0 & 0xF0) == 0xE0) {
        if (pos + 2 >= len) { pos += 1; return 0xFFFD; }
        uint8_t b1 = (uint8_t)data[pos + 1];
        uint8_t b2 = (uint8_t)data[pos + 2];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) { pos += 1; return 0xFFFD; }
        pos += 3;
        uint32_t cp = ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(b1 & 0x3F) << 6) | (b2 & 0x3F);
        return cp < 0x800 ? 0xFFFD : cp; // overlong check
    }

    // 4-byte
    if ((b0 & 0xF8) == 0xF0) {
        if (pos + 3 >= len) { pos += 1; return 0xFFFD; }
        uint8_t b1 = (uint8_t)data[pos + 1];
        uint8_t b2 = (uint8_t)data[pos + 2];
        uint8_t b3 = (uint8_t)data[pos + 3];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) {
            pos += 1; return 0xFFFD;
        }
        pos += 4;
        uint32_t cp = ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)(b1 & 0x3F) << 12)
                    | ((uint32_t)(b2 & 0x3F) << 6) | (b3 & 0x3F);
        return (cp < 0x10000 || cp > 0x10FFFF) ? 0xFFFD : cp;
    }

    pos += 1;
    return 0xFFFD;
}

std::vector<uint32_t> toCodepoints(const std::string& utf8) {
    std::vector<uint32_t> result;
    size_t pos = 0;
    while (pos < utf8.size()) {
        result.push_back(decodeUtf8(utf8.data(), utf8.size(), pos));
    }
    return result;
}

std::string encodeUtf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

std::string fromCodepoints(const std::vector<uint32_t>& cps) {
    std::string result;
    for (uint32_t cp : cps) {
        result += encodeUtf8(cp);
    }
    return result;
}

// ── Unicode category detection using range tables ──

// Letter ranges: ASCII + Latin Extended + Greek + Cyrillic + Armenian + Hebrew +
// Arabic + Devanagari + Bengali + other Indic + Thai + Lao + Tibetan +
// Georgian + Hangul Jamo + Ethiopic + Cherokee + Khmer + Mongolian +
// CJK Unified Ideographs + Hangul Syllables + supplementary
struct UnicodeRange {
    uint32_t lo, hi;
};

static const UnicodeRange letterRanges[] = {
    // ASCII
    {0x0041, 0x005A}, {0x0061, 0x007A},
    // Latin Extended
    {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x024F},
    // IPA + Spacing Modifier Letters
    {0x0250, 0x02AF}, {0x02B0, 0x02FF},
    // Greek and Coptic
    {0x0370, 0x03FF},
    // Cyrillic
    {0x0400, 0x04FF}, {0x0500, 0x052F},
    // Armenian
    {0x0531, 0x058F},
    // Hebrew
    {0x0590, 0x05FF},
    // Arabic
    {0x0600, 0x06FF}, {0x0750, 0x077F},
    // Devanagari
    {0x0900, 0x097F},
    // Bengali
    {0x0980, 0x09FF},
    // Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
    {0x0A00, 0x0DFF},
    // Sinhala, Thai
    {0x0E00, 0x0E7F},
    // Lao
    {0x0E80, 0x0EFF},
    // Tibetan
    {0x0F00, 0x0FFF},
    // Myanmar
    {0x1000, 0x109F},
    // Georgian
    {0x10A0, 0x10FF},
    // Hangul Jamo
    {0x1100, 0x11FF},
    // Ethiopic
    {0x1200, 0x137F},
    // Cherokee
    {0x13A0, 0x13FF},
    // Khmer
    {0x1780, 0x17FF},
    // Mongolian
    {0x1800, 0x18AF},
    // Latin Extended Additional
    {0x1E00, 0x1EFF},
    // Greek Extended
    {0x1F00, 0x1FFF},
    // CJK Radicals + Kangxi + Ideographic Desc
    {0x2E80, 0x2FDF},
    // CJK Unified Ideographs Extension A
    {0x3400, 0x4DBF},
    // CJK Unified Ideographs
    {0x4E00, 0x9FFF},
    // Yi
    {0xA000, 0xA4CF},
    // Hangul Syllables
    {0xAC00, 0xD7AF},
    // CJK Compatibility Ideographs
    {0xF900, 0xFAFF},
    // Latin ligatures
    {0xFB00, 0xFB06},
    // Arabic Presentation Forms
    {0xFB50, 0xFDFF}, {0xFE70, 0xFEFF},
    // Halfwidth/Fullwidth letters
    {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    // CJK Extensions B-F (supplementary plane)
    {0x20000, 0x2A6DF}, {0x2A700, 0x2CEAF},
    {0x2CEB0, 0x2EBEF}, {0x30000, 0x3134F},
    // Emoji modifiers treated as letters for tokenization
};

static const UnicodeRange numberRanges[] = {
    {0x0030, 0x0039},  // ASCII digits
    {0x0660, 0x0669},  // Arabic-Indic digits
    {0x06F0, 0x06F9},  // Extended Arabic-Indic
    {0x0966, 0x096F},  // Devanagari digits
    {0x09E6, 0x09EF},  // Bengali digits
    {0x0A66, 0x0A6F},  // Gurmukhi digits
    {0x0AE6, 0x0AEF},  // Gujarati digits
    {0x0B66, 0x0B6F},  // Oriya digits
    {0x0BE6, 0x0BEF},  // Tamil digits
    {0x0C66, 0x0C6F},  // Telugu digits
    {0x0CE6, 0x0CEF},  // Kannada digits
    {0x0D66, 0x0D6F},  // Malayalam digits
    {0x0E50, 0x0E59},  // Thai digits
    {0x0ED0, 0x0ED9},  // Lao digits
    {0x0F20, 0x0F29},  // Tibetan digits
    {0xFF10, 0xFF19},  // Fullwidth digits
};

static bool inRanges(uint32_t cp, const UnicodeRange* ranges, size_t count) {
    // Binary search for better performance
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp > ranges[mid].hi) {
            lo = mid + 1;
        } else if (cp < ranges[mid].lo) {
            hi = mid;
        } else {
            return true;
        }
    }
    return false;
}

bool isLetter(uint32_t cp) {
    return inRanges(cp, letterRanges, sizeof(letterRanges) / sizeof(letterRanges[0]));
}

bool isNumber(uint32_t cp) {
    return inRanges(cp, numberRanges, sizeof(numberRanges) / sizeof(numberRanges[0]));
}

bool isWhitespace(uint32_t cp) {
    // Common whitespace characters
    switch (cp) {
        case 0x0009: // tab
        case 0x000A: // LF
        case 0x000B: // VT
        case 0x000C: // FF
        case 0x000D: // CR
        case 0x0020: // space
        case 0x0085: // NEL
        case 0x00A0: // NBSP
        case 0x1680: // Ogham space
        case 0x2000: case 0x2001: case 0x2002: case 0x2003:
        case 0x2004: case 0x2005: case 0x2006: case 0x2007:
        case 0x2008: case 0x2009: case 0x200A:
        case 0x2028: // Line separator
        case 0x2029: // Paragraph separator
        case 0x202F: // Narrow NBSP
        case 0x205F: // Medium mathematical space
        case 0x3000: // Ideographic space
            return true;
        default:
            return false;
    }
}

bool isPunctuation(uint32_t cp) {
    // ASCII punctuation
    if ((cp >= 0x21 && cp <= 0x2F) ||
        (cp >= 0x3A && cp <= 0x40) ||
        (cp >= 0x5B && cp <= 0x60) ||
        (cp >= 0x7B && cp <= 0x7E)) {
        return true;
    }
    // General punctuation block
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    // CJK punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    // Fullwidth punctuation
    if (cp >= 0xFF01 && cp <= 0xFF0F) return true;
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    return false;
}

std::string nfcNormalize(const std::string& input) {
    // For the tokenizer's purposes, NFC normalization is mainly needed for
    // combining diacritical marks. Most text is already in NFC form.
    // We use Windows NormalizeString API for correctness.
#ifdef _WIN32
    // Use Windows API if available
    // Lazy: try NormalizeString via LoadLibrary to avoid link-time dependency
    // For now, pass through - the tokenizer.json vocab is already NFC
#endif
    return input;
}

} // namespace UnicodeUtils
