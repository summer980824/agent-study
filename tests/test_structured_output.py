"""``agent_study.structured_output`` 的单元测试。

``TaskSchema`` 是给 LLM 结构化输出用的契约，所以除了字段校验本身，
还要保证生成的 JSON Schema 真的把约束带出去（否则提示词里就没有约束了）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_study.structured_output import TaskSchema

VALID = {
    "title": "写周报",
    "description": "整理本周进展并同步给团队",
    "priority": "high",
    "tags": ["docs"],
    "estimated_minutes": 60,
}


def _valid(**overrides: object) -> TaskSchema:
    return TaskSchema(**{**VALID, **overrides})


# ---------------------------------------------------------------------------
# 正常构造
# ---------------------------------------------------------------------------


class TestValidTask:
    def test_accepts_a_complete_task(self):
        task = _valid()
        assert task.title == "写周报"
        assert task.description == "整理本周进展并同步给团队"
        assert task.priority == "high"
        assert task.tags == ["docs"]
        assert task.estimated_minutes == 60

    def test_tags_default_to_empty_list(self):
        task = TaskSchema(
            title="写周报",
            description="整理本周进展",
            priority="low",
            estimated_minutes=10,
        )
        assert task.tags == []

    def test_default_tags_are_not_shared_between_instances(self):
        first = TaskSchema(
            title="a", description="b", priority="low", estimated_minutes=1
        )
        first.tags.append("x")
        second = TaskSchema(
            title="a", description="b", priority="low", estimated_minutes=1
        )
        assert second.tags == []

    def test_round_trips_through_dict_and_json(self):
        task = _valid()
        assert TaskSchema.model_validate(task.model_dump()) == task
        assert TaskSchema.model_validate_json(task.model_dump_json()) == task


# ---------------------------------------------------------------------------
# title
# ---------------------------------------------------------------------------


class TestTitle:
    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            _valid(title="")

    def test_rejects_blank_string(self):
        with pytest.raises(ValidationError, match="cannot be blank"):
            _valid(title="   ")

    def test_rejects_over_100_characters(self):
        with pytest.raises(ValidationError):
            _valid(title="x" * 101)

    def test_accepts_exactly_100_characters(self):
        assert _valid(title="x" * 100).title == "x" * 100

    def test_strips_surrounding_whitespace(self):
        assert _valid(title="  写周报  ").title == "写周报"

    def test_length_limit_counts_the_raw_string(self):
        # 长度上限按 strip 之前的原始字符串计，和 JSON Schema 的 maxLength 口径一致
        assert _valid(title=" " * 10 + "x" * 90).title == "x" * 90

        with pytest.raises(ValidationError):
            _valid(title=" " * 10 + "x" * 100 + " " * 10)


# ---------------------------------------------------------------------------
# description
# ---------------------------------------------------------------------------


class TestDescription:
    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            _valid(description="")

    def test_rejects_blank_string(self):
        with pytest.raises(ValidationError, match="cannot be blank"):
            _valid(description=" \t\n ")

    def test_strips_surrounding_whitespace(self):
        assert _valid(description="  整理进展  ").description == "整理进展"

    def test_has_no_length_limit(self):
        assert _valid(description="长" * 500).description == "长" * 500


# ---------------------------------------------------------------------------
# priority
# ---------------------------------------------------------------------------


class TestPriority:
    @pytest.mark.parametrize("priority", ["low", "medium", "high"])
    def test_accepts_the_three_allowed_values(self, priority: str):
        assert _valid(priority=priority).priority == priority

    @pytest.mark.parametrize("priority", ["urgent", "LOW", "中", "", "none"])
    def test_rejects_anything_else(self, priority: str):
        with pytest.raises(ValidationError):
            _valid(priority=priority)

    def test_is_required(self):
        with pytest.raises(ValidationError):
            TaskSchema(
                title="a", description="b", estimated_minutes=10
            )


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_accepts_multiple_tags(self):
        assert _valid(tags=["docs", "weekly"]).tags == ["docs", "weekly"]

    def test_accepts_empty_list(self):
        assert _valid(tags=[]).tags == []

    def test_rejects_non_string_items(self):
        with pytest.raises(ValidationError):
            _valid(tags=["docs", 1])

    def test_rejects_a_bare_string(self):
        with pytest.raises(ValidationError):
            _valid(tags="docs")


# ---------------------------------------------------------------------------
# estimated_minutes
# ---------------------------------------------------------------------------


class TestEstimatedMinutes:
    @pytest.mark.parametrize("minutes", [1, 30, 480])
    def test_accepts_values_in_range(self, minutes: int):
        assert _valid(estimated_minutes=minutes).estimated_minutes == minutes

    def test_rejects_zero(self):
        with pytest.raises(ValidationError):
            _valid(estimated_minutes=0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            _valid(estimated_minutes=-5)

    def test_rejects_over_480(self):
        with pytest.raises(ValidationError):
            _valid(estimated_minutes=481)

    def test_rejects_non_integer(self):
        with pytest.raises(ValidationError):
            _valid(estimated_minutes=60.5)

    def test_is_required(self):
        with pytest.raises(ValidationError):
            TaskSchema(title="a", description="b", priority="low")


# ---------------------------------------------------------------------------
# JSON Schema（提示词就是靠它生成的）
# ---------------------------------------------------------------------------


class TestJsonSchema:
    def test_declares_all_required_fields(self):
        schema = TaskSchema.model_json_schema()
        assert set(schema["required"]) == {
            "title",
            "description",
            "priority",
            "estimated_minutes",
        }

    def test_carries_the_priority_enum(self):
        properties = TaskSchema.model_json_schema()["properties"]
        assert properties["priority"]["enum"] == ["low", "medium", "high"]

    def test_carries_the_numeric_and_length_limits(self):
        properties = TaskSchema.model_json_schema()["properties"]
        assert properties["title"]["maxLength"] == 100
        assert properties["title"]["minLength"] == 1
        assert properties["estimated_minutes"]["maximum"] == 480
        assert properties["estimated_minutes"]["exclusiveMinimum"] == 0

    def test_tags_is_an_array_of_strings(self):
        properties = TaskSchema.model_json_schema()["properties"]
        assert properties["tags"]["type"] == "array"
        assert properties["tags"]["items"]["type"] == "string"
