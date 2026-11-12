"""Batching-only model. No actual network, queues, headers or codec are modeled."""


# Purpose: Group captured events under one of three release policies.
# Inputs: Time-ordered events, policy ('time', 'count', 'hybrid'), time window,
#         maximum count, and observation horizon, all times in seconds.
# Method: 'time' uses global window boundaries. 'count' sends only a full batch.
#         'hybrid' sends at the count limit OR window seconds after its first event.
# Output: (packets, pending), where each packet is (release_time, event_list).
# Important: Never force-flush the tail at the horizon. This exposes starvation
#            of count-only batching when motion stops. Events exactly at a time
#            deadline enter a new batch. Empty packets are never generated.
def packetize(events, policy="time", window=0.02, count=700, horizon=4.0):
    if policy not in {"time", "count", "hybrid"}:
        raise ValueError("Unknown packet policy")
    if window <= 0 or count <= 0 or int(count) != count or horizon < 0:
        raise ValueError("Window/count must be positive; horizon nonnegative")
    import math
    packets, pending = [], []
    deadline = float("inf")
    previous = -float("inf")
    for event in events:
        timestamp = event[2]
        if timestamp < previous:
            raise ValueError("Events must be sorted by timestamp")
        previous = timestamp
        if timestamp > horizon:
            break
        if pending and policy != "count" and timestamp >= deadline:
            packets.append((deadline, pending))
            pending = []
        if not pending:
            deadline = ((math.floor(timestamp / window) + 1) * window
                        if policy == "time" else timestamp + window)
        pending.append(event)
        if policy != "time" and len(pending) >= count:
            packets.append((timestamp, pending))
            pending = []
    if pending and policy != "count" and deadline <= horizon:
        packets.append((deadline, pending))
        pending = []
    return packets, pending


# Purpose: Calculate batching delay without hiding events that were never sent.
# Input: The (packets,pending) result from packetize and the same horizon.
# Method: For each SENT event compute release_time - capture_time; average across
#         events, not packets. Track pending count and oldest pending age separately.
# Output: A JSON-compatible metrics dictionary; unavailable waits are None.
# Demo role: Low average delay among sent events cannot excuse an unsent tail.
def packet_statistics(result, horizon=4.0):
    packets, pending = result
    waits = [max(0.0, released - event[2])
             for released, batch in packets for event in batch]
    return {
        "packets": len(packets), "sent_events": len(waits),
        "pending_events": len(pending),
        "mean_wait_ms": 1000 * sum(waits) / len(waits) if waits else None,
        "max_wait_ms": 1000 * max(waits) if waits else None,
        "oldest_pending_ms": 1000 * (horizon - pending[0][2]) if pending else 0.0,
    }
