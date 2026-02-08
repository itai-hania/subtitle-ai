"""Unit tests for pure utility functions (Issue 9A)."""

import pytest

from processing import (
    format_timestamp,
    parse_ffmpeg_error,
    process_srt_for_rtl,
    srt_to_plain_text,
    subtitles_to_srt,
    wrap_rtl,
)
from app import validate_job_id


# ============================================
# format_timestamp
# ============================================

class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00,000"

    def test_one_second(self):
        assert format_timestamp(1.0) == "00:00:01,000"

    def test_with_milliseconds(self):
        assert format_timestamp(1.5) == "00:00:01,500"

    def test_one_minute(self):
        assert format_timestamp(60.0) == "00:01:00,000"

    def test_one_hour(self):
        assert format_timestamp(3600.0) == "01:00:00,000"

    def test_complex_time(self):
        # 1h 23m 45.678s
        assert format_timestamp(5025.678) == "01:23:45,678"

    def test_sub_millisecond(self):
        # 0.0001 seconds — should round to 0 milliseconds
        result = format_timestamp(0.0001)
        assert result == "00:00:00,000"

    def test_large_value(self):
        # 10 hours
        result = format_timestamp(36000.0)
        assert result == "10:00:00,000"

    def test_just_under_a_second(self):
        result = format_timestamp(0.999)
        assert result == "00:00:00,999"


# ============================================
# validate_job_id
# ============================================

class TestValidateJobId:
    def test_valid_uuid_short(self):
        assert validate_job_id("abc12345") is True

    def test_valid_with_hyphens(self):
        assert validate_job_id("a1b2c3d4-e5f6") is True

    def test_full_uuid(self):
        assert validate_job_id("550e8400-e29b-41d4-a716-44665544") is True

    def test_empty_string(self):
        assert validate_job_id("") is False

    def test_path_traversal(self):
        assert validate_job_id("../../etc") is False
        assert validate_job_id("../passwd") is False

    def test_special_characters(self):
        assert validate_job_id("abc;rm -rf /") is False
        assert validate_job_id("test<script>") is False

    def test_too_long(self):
        assert validate_job_id("a" * 37) is False

    def test_max_length(self):
        assert validate_job_id("a" * 36) is True

    def test_single_char(self):
        assert validate_job_id("a") is True

    def test_only_numbers(self):
        assert validate_job_id("12345678") is True

    def test_underscores_rejected(self):
        assert validate_job_id("test_id") is False

    def test_spaces_rejected(self):
        assert validate_job_id("test id") is False


# ============================================
# parse_ffmpeg_error
# ============================================

class TestParseFfmpegError:
    def test_disk_full(self):
        result = parse_ffmpeg_error("No space left on device")
        assert "disk is full" in result.lower()

    def test_invalid_data(self):
        result = parse_ffmpeg_error("Invalid data found when processing input")
        assert "corrupt" in result.lower() or "unsupported" in result.lower()

    def test_codec_not_supported(self):
        result = parse_ffmpeg_error("Codec not currently supported in container")
        assert "codec" in result.lower()

    def test_decoder_not_found(self):
        result = parse_ffmpeg_error("Decoder h265 not found")
        assert "codec" in result.lower()

    def test_permission_denied(self):
        result = parse_ffmpeg_error("Permission denied")
        assert "permission" in result.lower()

    def test_no_such_file(self):
        result = parse_ffmpeg_error("No such file or directory")
        assert "not found" in result.lower()

    def test_no_stream(self):
        result = parse_ffmpeg_error("does not contain any stream")
        assert "stream" in result.lower()

    def test_unknown_error(self):
        result = parse_ffmpeg_error("some random unknown error message")
        assert "Video processing failed" in result

    def test_empty_stderr(self):
        result = parse_ffmpeg_error("")
        assert "Unknown error" in result

    def test_none_stderr(self):
        result = parse_ffmpeg_error(None)
        assert "Unknown error" in result

    def test_long_stderr_truncated(self):
        long_msg = "x" * 500
        result = parse_ffmpeg_error(long_msg)
        assert len(result) < 300  # Should be truncated


# ============================================
# wrap_rtl / process_srt_for_rtl
# ============================================

class TestRtl:
    def test_wrap_rtl_basic(self):
        result = wrap_rtl("שלום")
        assert result.startswith("\u202B")
        assert result.endswith("\u202C")
        assert "שלום" in result

    def test_wrap_rtl_empty(self):
        result = wrap_rtl("")
        assert result == "\u202B\u202C"

    def test_process_srt_for_rtl(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nשלום עולם\n\n2\n00:00:04,000 --> 00:00:06,000\nלהתראות"
        result = process_srt_for_rtl(srt)
        # Text lines should have RTL markers
        assert "\u202B" in result
        assert "\u202C" in result
        # Timing should be preserved
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "00:00:04,000 --> 00:00:06,000" in result

    def test_process_srt_for_rtl_preserves_structure(self):
        srt = "1\n00:00:00,000 --> 00:00:01,000\nHello"
        result = process_srt_for_rtl(srt)
        blocks = result.strip().split("\n\n")
        assert len(blocks) == 1
        lines = blocks[0].split("\n")
        assert lines[0] == "1"
        assert "-->" in lines[1]


# ============================================
# subtitles_to_srt
# ============================================

class TestSubtitlesToSrt:
    def test_basic(self):
        subs = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 2, "start": 3.0, "end": 5.0, "text": "World"},
        ]
        result = subtitles_to_srt(subs)
        assert "1\n00:00:00,000 --> 00:00:02,000\nHello\n" in result
        assert "2\n00:00:03,000 --> 00:00:05,000\nWorld\n" in result

    def test_empty_list(self):
        assert subtitles_to_srt([]) == ""

    def test_single_subtitle(self):
        subs = [{"id": 1, "start": 1.5, "end": 3.5, "text": "Test"}]
        result = subtitles_to_srt(subs)
        assert "00:00:01,500 --> 00:00:03,500" in result
        assert "Test" in result


# ============================================
# srt_to_plain_text
# ============================================

class TestSrtToPlainText:
    def test_basic(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nGoodbye"
        result = srt_to_plain_text(srt)
        assert result == "Hello world\nGoodbye"

    def test_empty(self):
        assert srt_to_plain_text("") == ""

    def test_multiline_subtitle(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two"
        result = srt_to_plain_text(srt)
        assert result == "Line one\nLine two"

    def test_strips_only_timing_and_number(self):
        srt = "1\n00:00:00,000 --> 00:00:01,000\nKeep this\n\n2\n00:00:02,000 --> 00:00:03,000\nAnd this"
        result = srt_to_plain_text(srt)
        assert "Keep this" in result
        assert "And this" in result
        assert "-->" not in result
