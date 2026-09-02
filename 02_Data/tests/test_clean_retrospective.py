"""Checks for clean_retrospective.py: SRT parsing, rolling-window dedup,
repetition metric, transcript classification, and the rebuild policy. No
network, no real dataset -- everything runs on a synthetic video dir.

    cd 02_Data && ../.venv/bin/python tests/test_clean_retrospective.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clean_retrospective as cr  # noqa: E402

SRT_ROLLING = """1
00:00:00,000 --> 00:00:02,000
hello there everyone

2
00:00:02,000 --> 00:00:04,000
hello there everyone
welcome to the show

3
00:00:04,000 --> 00:00:06,000
welcome to the show
today we build a chair

4
00:00:06,000 --> 00:00:08,000
today we build a chair
"""


def _mkvideo(tmp, source="auto_captions", srt=SRT_ROLLING, transcript="x y z",
             had_auto=True):
    d = os.path.join(tmp, "cat", "vid00000001")
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    open(os.path.join(d, ".done"), "w").close()
    if srt is not None:
        with open(os.path.join(d, "captions.en.srt"), "w", encoding="utf-8") as f:
            f.write(srt)
    if transcript is not None:
        with open(os.path.join(d, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)
    with open(os.path.join(d, "transcript_info.json"), "w", encoding="utf-8") as f:
        json.dump({"source": source, "had_auto_captions": had_auto}, f)
    return d


def test_srt_parse_drops_index_and_time_lines_by_shape():
    tmp = tempfile.mkdtemp()
    try:
        d = _mkvideo(tmp)
        lines = cr.parse_srt_text_lines(os.path.join(d, "captions.en.srt"))
        assert "1" not in lines and not any("-->" in l for l in lines)
        assert lines[0] == "hello there everyone"
        # markup is stripped
        with open(os.path.join(d, "captions.en.srt"), "a", encoding="utf-8") as f:
            f.write("\n5\n00:00:08,000 --> 00:00:09,000\n<i>styled</i> text\n")
        assert cr.parse_srt_text_lines(os.path.join(d, "captions.en.srt"))[-1] == "styled text"
    finally:
        shutil.rmtree(tmp)


def test_rolling_dedup_collapses_duplicates_and_progressive_cues():
    lines = ["hello there everyone", "hello there everyone", "welcome to the show",
             "welcome to the show", "today we build a chair", "today we build a chair"]
    assert cr.dedup_rolling_lines(lines) == [
        "hello there everyone", "welcome to the show", "today we build a chair"]
    # progressive cue: previous kept line is a prefix of the new one -> replaced
    assert cr.dedup_rolling_lines(["hello", "hello world"]) == ["hello world"]
    # previous line ENDS with the new one on a WORD boundary (tail repeat) -> dropped
    assert cr.dedup_rolling_lines(["a b c", "c"]) == ["a b c"]
    # ... but a mid-word suffix/prefix is a DIFFERENT word and must survive
    assert cr.dedup_rolling_lines(["solution", "on"]) == ["solution", "on"]
    assert cr.dedup_rolling_lines(["This", "is"]) == ["This", "is"]
    assert cr.dedup_rolling_lines(["hel", "hello world"]) == ["hel", "hello world"]
    assert cr.dedup_rolling_lines([]) == []


def test_dup8_ratio_flags_triplicated_text_and_passes_clean_text():
    clean = ("w%d " % i for i in range(100))
    assert cr.dup8_ratio("".join(clean).split()) == 0.0
    tri = ("one two three four five six seven eight " * 3).split()
    assert cr.dup8_ratio(tri) > 0.5
    assert cr.dup8_ratio("a b".split()) == 0.0  # <9 words: defined as 0


def test_script_profile_separates_english_from_devanagari():
    latin, stop = cr.script_profile("the cat sat on the mat and it was good")
    assert latin == 1.0 and stop > 0.3
    latin_hi, _ = cr.script_profile("नमस्ते दुनिया")
    assert latin_hi == 0.0


def test_classify_transcript_kinds_and_precedence():
    # varied sentences: enough words and stopwords WITHOUT tripping the dup8 gate
    en = " ".join(f"the cat number {i} sat on a mat and it was good" for i in range(10))
    # native English
    kind, usable, *_ = cr.classify_transcript(en, "auto_captions", "en", False)
    assert kind == "native_en" and usable
    # declared hi + auto captions in English = machine translation
    kind, usable, *_ = cr.classify_transcript(en, "auto_captions", "hi", False)
    assert kind == "translated_en" and usable
    # declared hi + whisper = hallucinated English, regardless of fluency
    kind, usable, *_ = cr.classify_transcript(en, "whisper", "hi", False)
    assert kind == "whisper_on_non_english" and not usable
    # sparse text wins over everything except missing
    kind, usable, *_ = cr.classify_transcript("hi there", "whisper", "hi", False)
    assert kind == "sparse" and not usable
    # missing
    kind, usable, *_ = cr.classify_transcript(None, "whisper", "en", False)
    assert kind == "missing" and not usable
    # no_speech vetoes usability but not the kind
    kind, usable, *_ = cr.classify_transcript(en, "auto_captions", "en", True)
    assert kind == "native_en" and not usable
    # repetition-dominated text is not usable
    loop = "the cat sat on the mat and it was good " * 40
    kind, usable, *_ = cr.classify_transcript(loop, "auto_captions", "en", False)
    assert kind == "native_en" and not usable


def test_rebuild_policy_auto_rebuilds_whisper_only_salvaged_when_sparse():
    tmp = tempfile.mkdtemp()
    try:
        # auto-captions source: always rebuilt from srt
        d = _mkvideo(tmp, source="auto_captions", transcript="a a a")
        assert cr.rebuild_transcript(d) == "rebuilt"
        clean = open(os.path.join(d, cr.CLEAN_NAME), encoding="utf-8").read()
        assert clean.splitlines() == ["hello there everyone", "welcome to the show",
                                      "today we build a chair"]
        # whisper source with a healthy transcript: srt NOT used
        long_txt = "word " * 100
        d2 = _mkvideo(os.path.join(tmp, "2"), source="whisper", transcript=long_txt)
        assert cr.rebuild_transcript(d2) is None
        assert not os.path.exists(os.path.join(d2, cr.CLEAN_NAME))
        # whisper source, sparse output, unused srt: salvaged
        d3 = _mkvideo(os.path.join(tmp, "3"), source="whisper", transcript="uh")
        assert cr.rebuild_transcript(d3) == "salvaged"
        # whisper source, transcript MISSING, srt present: salvaged
        d4 = _mkvideo(os.path.join(tmp, "4"), source="whisper", transcript=None)
        assert cr.rebuild_transcript(d4) == "salvaged"
        # no srt at all: nothing to do
        d5 = _mkvideo(os.path.join(tmp, "5"), source="whisper", srt=None, transcript="uh")
        assert cr.rebuild_transcript(d5) is None
    finally:
        shutil.rmtree(tmp)


def test_iso8601_duration_parse():
    assert cr._iso8601_duration_sec("PT4M10S") == 250
    assert cr._iso8601_duration_sec("PT1H2M3S") == 3723
    assert cr._iso8601_duration_sec("PT45S") == 45
    assert cr._iso8601_duration_sec("P1DT2H") == 93600
    assert cr._iso8601_duration_sec("") is None and cr._iso8601_duration_sec(None) is None


def test_tag_token_ratio_is_token_based_not_line_based():
    """SRT text is one line per cue, whisper text is one paragraph -- a
    per-line ratio compared two different things and never fired."""
    assert cr.tag_token_ratio("[Music] hello world") == 1 / 3  # 1 tag token of 3
    assert cr.tag_token_ratio("[Music] [Applause] hello world") == 0.5
    assert cr.tag_token_ratio("hello world") == 0.0
    assert cr.tag_token_ratio("") == 0.0
    # same content, different line structure -> same ratio
    assert cr.tag_token_ratio("[Music]\nhello world") == cr.tag_token_ratio("[Music] hello world")


# --- regressions for the 2026-08-31 review findings -------------------------

def test_merge_extra_is_atomic_and_refuses_to_overwrite_corrupt_json():
    """A crash mid-write must not truncate the file, and a file that exists but
    does not parse must stop the run instead of being merged into {} (which
    would discard the irreplaceable backfill fields)."""
    tmp = tempfile.mkdtemp()
    try:
        d = _mkvideo(tmp)
        path = os.path.join(d, cr.EXTRA_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"published_at_utc": "2026-01-01T00:00:00Z", "status": "ok"}, f)
        cr._merge_extra(d, {"is_short": "false"})
        got = json.load(open(path, encoding="utf-8"))
        assert got["published_at_utc"] == "2026-01-01T00:00:00Z" and got["is_short"] == "false"
        assert not os.path.exists(path + ".tmp")  # temp file cleaned up by rename

        with open(path, "w", encoding="utf-8") as f:
            f.write('{"published_at_utc": "2026-01-0')  # truncated fragment
        try:
            cr._merge_extra(d, {"is_short": "true"})
            raise AssertionError("expected a fatal exit on unparseable metadata_extra.json")
        except SystemExit:
            pass
        assert open(path, encoding="utf-8").read() == '{"published_at_utc": "2026-01-0'
    finally:
        shutil.rmtree(tmp)


def test_atomic_write_leaves_old_content_on_failure():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "t.txt")
        cr._atomic_write_text(p, "original")
        try:
            cr._atomic_write_text(p, object())  # write() raises mid-call
        except TypeError:
            pass
        assert open(p, encoding="utf-8").read() == "original"
    finally:
        shutil.rmtree(tmp)


def test_salvaged_srt_is_classified_as_captions_not_whisper():
    """A whisper-source video salvaged from its SRT carries auto-caption text;
    judging it by the stale whisper source mislabels it hallucinated English."""
    en = " ".join(f"the cat number {i} sat on a mat and it was good" for i in range(10))
    kind, usable, *_ = cr.classify_transcript(en, "auto_captions", "hi", False)
    assert kind == "translated_en" and usable
    kind, _, *_ = cr.classify_transcript(en, "whisper", "hi", False)
    assert kind == "whisper_on_non_english"


def test_speech_veto_only_at_the_high_threshold():
    """pause_ratio 0.5-0.8 is normal in long clean speech; only >=0.8 vetoes."""
    en = " ".join(f"the cat number {i} sat on a mat and it was good" for i in range(10))
    assert cr.NO_SPEECH_VETO > cr.NO_SPEECH_PAUSE_RATIO
    _, usable_mid, *_ = cr.classify_transcript(en, "auto_captions", "en", False)
    _, usable_veto, *_ = cr.classify_transcript(en, "auto_captions", "en", True)
    assert usable_mid and not usable_veto


def test_age_days_survives_null_dates():
    """metadata.json with null upload_date must blank the column, not crash."""
    tmp = tempfile.mkdtemp()
    try:
        d = _mkvideo(tmp)
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"channel_id": "c", "upload_date": None, "collected_at": None,
                       "view_count": 5}, f)
        with open(os.path.join(d, cr.EXTRA_NAME), "w", encoding="utf-8") as f:
            json.dump({"status": "ok"}, f)
        rows = cr.build_manifest(tmp)
        assert len(rows) == 1 and rows[0]["age_days_at_collection"] == ""
        assert rows[0]["duration_source"] == "none" and rows[0]["subs_source"] == "none"
    finally:
        shutil.rmtree(tmp)


def test_numeric_caption_text_survives_but_index_lines_do_not():
    """A cue whose text is just a number (stitch counts, prices, model
    numbers) must not be mistaken for an SRT index line."""
    tmp = tempfile.mkdtemp()
    try:
        srt = ("1\n00:00:00,000 --> 00:00:02,000\nthe price is\n\n"
               "2\n00:00:02,000 --> 00:00:04,000\n2000\n\n"
               "3\n00:00:04,000 --> 00:00:06,000\ndollars\n\n"
               "75\n00:01:47,040 --> 00:01:48,710\n3\n4\n")
        d = _mkvideo(tmp, srt=srt)
        lines = cr.parse_srt_text_lines(os.path.join(d, "captions.en.srt"))
        assert lines == ["the price is", "2000", "dollars", "3", "4"]
    finally:
        shutil.rmtree(tmp)


def test_hard_space_escape_and_speaker_prefix_and_tag_lines_are_normalised():
    tmp = tempfile.mkdtemp()
    try:
        srt = ("1\n00:00:00,000 --> 00:00:02,000\n>> hello\\hthere\n\n"
               "2\n00:00:02,000 --> 00:00:04,000\n[Music]\n\n"
               "3\n00:00:04,000 --> 00:00:06,000\n>>>> friend\n")
        d = _mkvideo(tmp, srt=srt)
        lines = cr.parse_srt_text_lines(os.path.join(d, "captions.en.srt"))
        assert lines == ["hello there", "friend"]  # \h -> space, >> gone, [Music] dropped
        assert cr.parse_srt_text_lines(os.path.join(d, "captions.en.srt"),
                                       keep_tags=True) == ["hello there", "[Music]", "friend"]
    finally:
        shutil.rmtree(tmp)


def test_bracket_tags_are_excluded_from_word_and_language_statistics():
    """[Music] is not a word: counting it inflated transcript_words and
    depressed the English stopword rate."""
    text = "[Music] [Applause] the cat sat on the mat and it was good " * 6
    kind, _, n_words, _, _, stop, tags = cr.classify_transcript(text, "auto_captions", "en", False)
    assert n_words == 60          # 10 content words x 6 -- the 12 tag tokens excluded
    assert 0.15 < tags < 0.2      # ...but still reported, as 12 of 72 tokens
    assert kind == "native_en" and stop > 0.3


def test_read_json_distinguishes_absent_from_corrupt():
    tmp = tempfile.mkdtemp()
    try:
        assert cr._read_json(os.path.join(tmp, "nope.json")) is None
        p = os.path.join(tmp, "bad.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"a": ')
        try:
            cr._read_json(p)
            raise AssertionError("corrupt JSON must raise, not read as absent")
        except cr.CorruptJSON:
            pass
    finally:
        shutil.rmtree(tmp)


def test_manifest_records_a_corrupt_json_instead_of_hiding_it():
    tmp = tempfile.mkdtemp()
    try:
        d = _mkvideo(tmp)
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"channel_id": "c", "view_count": 1, "duration": 30,
                       "channel_follower_count": 10, "upload_date": "20260101",
                       "collected_at": "2026-03-01T00:00:00Z"}, f)
        with open(os.path.join(d, cr.EXTRA_NAME), "w", encoding="utf-8") as f:
            f.write('{"status": "o')  # truncated
        row = cr.build_manifest(tmp)[0]
        assert row["flag_unreadable_json"] == cr.EXTRA_NAME
        assert row["api_status"] == "" and row["duration_sec"] == 30
    finally:
        shutil.rmtree(tmp)


def test_manifest_flags_a_truncated_clean_transcript_shadowing_raw():
    tmp = tempfile.mkdtemp()
    try:
        d = _mkvideo(tmp, source="auto_captions", transcript="word " * 200)
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"channel_id": "c", "upload_date": "20260101",
                       "collected_at": "2026-03-01T00:00:00Z", "view_count": 5,
                       "duration": 60, "channel_follower_count": 100}, f)
        with open(os.path.join(d, cr.EXTRA_NAME), "w", encoding="utf-8") as f:
            json.dump({"status": "ok"}, f)
        cr._atomic_write_text(os.path.join(d, cr.CLEAN_NAME), "three word fragment")
        row = cr.build_manifest(tmp)[0]
        assert row["flag_clean_shadows_raw"] == 1 and row["transcript_raw_words"] == 200
        # a legitimate 3x dedup (ratio ~0.35) must NOT trip the flag
        cr._atomic_write_text(os.path.join(d, cr.CLEAN_NAME), "word " * 70)
        assert cr.build_manifest(tmp)[0]["flag_clean_shadows_raw"] == 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} OK")
    print("ALL CLEANING CHECKS PASSED")
