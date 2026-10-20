import re


_CLOCK_TIME = (
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<period>a\.?m\.?|p\.?m\.?)"
)
_CURRENT_TIME_RE = re.compile(
    rf"\b(?:it's|it is|now is|right now it's)\s+{_CLOCK_TIME}\b", re.IGNORECASE)
_APPOINTMENT_TIME_RE = re.compile(
    rf"\bappointment\b.{{0,50}}?\bat\s+{_CLOCK_TIME}\b", re.IGNORECASE)
_TRAVEL_DURATION_RE = re.compile(
    r"\b(?:travel|drive|trip|commute)\b.{0,40}?\b(?:takes?|is)\s+"
    r"(?P<amount>\d+)\s*(?P<unit>minutes?|mins?|hours?|hrs?)\b", re.IGNORECASE)
_PLANNING_RE = re.compile(r"\b(?:schedule|plan)\b", re.IGNORECASE)


def _minutes(match: re.Match) -> int:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    period = match.group("period").casefold()[0]
    hour = hour % 12 + (12 if period == "p" else 0)
    return hour * 60 + minute


def _clock(minutes: int) -> str:
    hour, minute = divmod(minutes % (24 * 60), 60)
    period = "AM" if hour < 12 else "PM"
    hour = hour % 12 or 12
    return f"{hour}:{minute:02d} {period}"


def appointment_schedule(question: str) -> str | None:
    """Create a safe appointment plan or ask for the missing required time."""
    if not _PLANNING_RE.search(question) or "appointment" not in question.casefold():
        return None
    current = _CURRENT_TIME_RE.search(question)
    appointment = _APPOINTMENT_TIME_RE.search(question)
    travel = _TRAVEL_DURATION_RE.search(question)
    if not appointment:
        return "What time does your appointment start?"
    if not travel or int(travel.group("amount")) <= 0:
        return "How long does it take to travel one way to the appointment?"

    appointment_at = _minutes(appointment)
    amount = int(travel.group("amount"))
    duration = amount * (60 if travel.group("unit").casefold().startswith(("hour", "hr")) else 1)
    arrival = appointment_at - 15
    departure = arrival - duration

    lines = ["Let's protect your appointment:"]
    if current:
        now = _minutes(current)
        prep_end = max(now, departure - 15)
        if now < departure and prep_end > now:
            lines.append(
                f"- {_clock(now)}-{_clock(prep_end)}: Make a grocery list and get ready at your own pace.")
        if now < departure and departure > prep_end:
            lines.append(
                f"- {_clock(prep_end)}-{_clock(departure)}: Keep this time open as a buffer before leaving.")
        if now >= departure:
            lines.append(
                f"- The planned departure time was {_clock(departure)}. Leave as soon as you can; "
                "I can't tell whether the 15-minute arrival buffer is still possible.")
    lines.append(
        f"- {_clock(departure)}: Leave for the appointment ({amount} {travel.group('unit')} travel).")
    lines.append(f"- {_clock(arrival)}: Arrive, 15 minutes before the appointment.")
    if re.search(r"\b(grocer(?:y|ies)|shopping|shop)\b", question, re.IGNORECASE):
        lines.append(
            "- After the appointment: Get groceries. The appointment end time and shopping duration are unknown.")
    return "\n".join(lines)
