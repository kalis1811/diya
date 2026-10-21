import pytest

from brain.caretaker import (
    AutonomyLevel,
    CaretakerLedger,
    InvalidTransition,
    Task,
    TaskStatus,
)


def build_task(**overrides):
    values = {
        "objective": "Check service health",
        "scope": "Configured local service only",
        "planned_steps": ("health_check",),
        "autonomy": AutonomyLevel.OBSERVE,
    }
    values.update(overrides)
    return Task(**values)


@pytest.mark.parametrize("level", [AutonomyLevel.OBSERVE, AutonomyLevel.RECOMMEND])
def test_low_autonomy_never_executes_even_when_approval_is_claimed(level):
    assert not level.permits(user_approved=True)
    assert not level.permits(consequential=True, user_approved=True)


def test_consequential_action_needs_supervision_and_approval():
    assert not AutonomyLevel.ROUTINE.permits(consequential=True, user_approved=True)
    assert not AutonomyLevel.SUPERVISED.permits(consequential=True)
    assert AutonomyLevel.SUPERVISED.permits(consequential=True, user_approved=True)


def test_completion_requires_verification_and_terminal_states_cannot_resume():
    task = build_task()
    task.transition(TaskStatus.RUNNING)
    task.record_step("health_check")
    with pytest.raises(ValueError, match="verification evidence"):
        task.transition(TaskStatus.COMPLETED)
    task.transition(TaskStatus.COMPLETED, verification_evidence=("health endpoint returned 200",))
    with pytest.raises(InvalidTransition):
        task.transition(TaskStatus.RUNNING)


def test_approval_required_task_needs_explicit_confirmation_to_resume():
    task = build_task()
    task.transition(TaskStatus.APPROVAL_REQUIRED)
    with pytest.raises(PermissionError):
        task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.RUNNING, user_approved=True)


def test_steps_must_be_planned_and_cannot_repeat():
    task = build_task()
    task.transition(TaskStatus.RUNNING)
    with pytest.raises(ValueError, match="not part of"):
        task.record_step("unplanned")
    task.record_step("health_check")
    with pytest.raises(ValueError, match="already recorded"):
        task.record_step("health_check")


def test_json_ledger_persists_state_and_metadata_only_events(tmp_path):
    path = tmp_path / "tasks.json"
    ledger = CaretakerLedger(path)
    task = ledger.create_task(build_task(objective="private task text"))
    ledger.transition(task.task_id, TaskStatus.RUNNING)
    ledger.record_step(task.task_id, "health_check")
    ledger.transition(
        task.task_id,
        TaskStatus.COMPLETED,
        verification_evidence=("verified locally",),
    )

    restored = CaretakerLedger(path)
    saved_task = restored.get_task(task.task_id)
    assert saved_task.status is TaskStatus.COMPLETED
    assert saved_task.verification_evidence == ("verified locally",)
    assert [event.event for event in restored.events] == [
        "task_created", "status:running", "step_recorded", "status:completed"
    ]
    assert "private task text" not in path.read_text(encoding="utf-8").split('"events"')[1]


def test_policy_ceiling_can_reduce_but_never_increase_task_autonomy():
    ledger = CaretakerLedger()
    task = ledger.create_task(build_task(autonomy=AutonomyLevel.SUPERVISED))
    assert not ledger.authorize_action(
        task.task_id,
        consequential=True,
        user_approved=True,
        policy_level=AutonomyLevel.OBSERVE,
    )
    assert ledger.authorize_action(
        task.task_id,
        consequential=True,
        user_approved=True,
        policy_level=AutonomyLevel.SUPERVISED,
    )
    assert [event.event for event in ledger.events[-2:]] == [
        "action_denied", "action_authorized"
    ]


def test_task_can_be_deleted_with_its_audit_history(tmp_path):
    ledger = CaretakerLedger(tmp_path / "tasks.json")
    task = ledger.create_task(build_task())
    assert ledger.delete_task(task.task_id)
    assert not ledger.events
    assert CaretakerLedger(tmp_path / "tasks.json").list_tasks() == ()


def test_failed_persistence_rolls_back_task_and_audit_mutation(tmp_path):
    ledger = CaretakerLedger()
    task = ledger.create_task(build_task())
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    ledger.path = blocker / "tasks.json"

    with pytest.raises(OSError):
        ledger.transition(task.task_id, TaskStatus.RUNNING)

    assert ledger.get_task(task.task_id).status is TaskStatus.PENDING
    assert [event.event for event in ledger.events] == ["task_created"]


@pytest.mark.parametrize("values", [
    {"objective": " "},
    {"scope": " "},
    {"planned_steps": ("duplicate", "duplicate")},
    {"planned_steps": ("a", "b"), "max_steps": 1},
    {"max_steps": 51},
])
def test_invalid_task_definitions_are_rejected(values):
    with pytest.raises(ValueError):
        build_task(**values)
