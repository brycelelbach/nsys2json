import sqlite3
import argparse
import json
from pathlib import Path
import re
import os

from collections import defaultdict

_PID_TO_DEVICE = {}

# Code adapted from https://github.com/ezyang/nvprof2json

def parse_args():
    parser = argparse.ArgumentParser(description='Convert nsight systems sqlite output to Google Event Trace compatible JSON.')
    parser.add_argument("-f", '--filenames', help="Path to the input sqlite file(s).", required=True, nargs="+")
    parser.add_argument("-o", "--output", help="Output file name, default to same as input with .json extension.")
    parser.add_argument("-t", "--activity-type", help="Type of activities shown. Default to all.", default=["kernel", "nvtx-kernel"], choices=['kernel', 'nvtx', "nvtx-kernel", "cuda-api"], nargs="+")
    parser.add_argument("--nvtx-event-prefix", help="Filter NVTX events by their names' prefix.", type=str, nargs="*")
    parser.add_argument("--nvtx-color-scheme", help="""Color scheme for NVTX events.
                                                    Accepts a dict mapping a string to one of chrome tracing colors.
                                                    Events with names containing the string will be colored.
                                                    E.g. {"send": "thread_state_iowait", "recv": "thread_state_iowait", "compute": "thread_state_running"}
                                                    For details of the color scheme, see links in https://github.com/google/perfetto/issues/208
                                                    """, type=json.loads, default={})
    parser.add_argument("--align-traces", help="The event name based on which to align traces if multiple sqlite files are supplied. "
                                                "The event should be present in all files and occur at the same time.", type=str)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(args.filenames[0]).with_suffix(".json")
    return args

class ActivityType:
    KERNEL = "kernel"
    NVTX_CPU = "nvtx"
    NVTX_KERNEL = "nvtx-kernel"
    CUDA_API = "cuda-api"

def munge_time(t):
    """Take a timestamp from nsys (ns) and convert it into us (the default for chrome://tracing)."""
    # For strict correctness, divide by 1000, but this reduces accuracy.
    return t / 1000.

# For reference of the schema, see
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html#exporter-sqlite-schema

def parse_cupti_kernel_events(conn: sqlite3.Connection, strings: dict, filename=""):
    """
    Copied from the docs:
    CUPTI_ACTIVITY_KIND_KERNEL
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KRENEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER                                -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    """
    per_device_kernel_rows = defaultdict(list)
    per_device_kernel_events = defaultdict(list)
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        per_device_kernel_rows[row["deviceId"]].append(row)
        event = {
                "name": strings[row["shortName"]],
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": munge_time(row["start"]),
                "dur": munge_time(row["end"] - row["start"]),
                "tid": "Stream {}".format(row["streamId"]),
                "pid": "{} Device {}".format(filename, row["deviceId"]),
                "args": {
                    # TODO: More
                    },
                }
        per_device_kernel_events[row["deviceId"]].append(event)
    return per_device_kernel_rows, per_device_kernel_events

def link_pid_with_devices(conn: sqlite3.Connection, filename=""):
    # map each pid to a device. assumes each pid is associated with a single device
    global _PID_TO_DEVICE
    if filename not in _PID_TO_DEVICE:
        pid_to_device = {}
        for row in conn.execute("SELECT DISTINCT deviceId, globalPid / 0x1000000 % 0x1000000 AS PID FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            assert row["PID"] not in pid_to_device, \
                f"A single PID ({row['PID']}) is associated with multiple devices ({pid_to_device[row['PID']]} and {row['deviceId']})."
            pid_to_device[row["PID"]] = row["deviceId"]
        _PID_TO_DEVICE[filename] = pid_to_device
    return _PID_TO_DEVICE[filename]

def parse_nvtx_events(conn: sqlite3.Connection, event_prefix=None, color_scheme={}, filename=""):
    """
    Copied from the docs:
    NVTX_EVENTS
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- NVTX event type enum value. See docs for specifics.
    rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
    category                    INTEGER,                               -- User-controlled ID that can be used to group events.
    color                       INTEGER,                               -- Encoded ARGB color value.
    text                        TEXT,                                  -- Optional text message for non registered strings.
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId. See docs for specifics.
    textId                      INTEGER   REFERENCES StringIds(id),    -- StringId of the NVTX domain registered string.
    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    uint64Value                 INTEGER,                               -- One of possible payload value union members.
    int64Value                  INTEGER,                               -- One of possible payload value union members.
    doubleValue                 REAL,                                  -- One of possible payload value union members.
    uint32Value                 INTEGER,                               -- One of possible payload value union members.
    int32Value                  INTEGER,                               -- One of possible payload value union members.
    floatValue                  REAL,                                  -- One of possible payload value union members.
    jsonTextId                  INTEGER,                               -- One of possible payload value union members.
    jsonText                    TEXT                                   -- One of possible payload value union members.

    NVTX_EVENT_TYPES
    33 - NvtxCategory
    34 - NvtxMark
    39 - NvtxThread
    59 - NvtxPushPopRange
    60 - NvtxStartEndRange
    75 - NvtxDomainCreate
    76 - NvtxDomainDestroy
    """

    if event_prefix is None:
        match_text = ''
    else:
        match_text = " AND "
        if len(event_prefix) == 1:
            match_text += f"NVTX_EVENTS.text LIKE '{event_prefix[0]}%'"
        else:
            match_text += "("
            for idx, prefix in enumerate(event_prefix):
                match_text += f"NVTX_EVENTS.text LIKE '{prefix}%'"
                if idx == len(event_prefix) - 1:
                    match_text += ")"
                else:
                    match_text += " OR "

    per_device_nvtx_rows = defaultdict(list)
    per_device_nvtx_events = defaultdict(list)
    pid_to_device = link_pid_with_devices(conn)
    # eventType 59 is NvtxPushPopRange, which corresponds to torch.cuda.nvtx.range apis
    for row in conn.execute(f"SELECT start, end, text, globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID FROM NVTX_EVENTS WHERE NVTX_EVENTS.eventType == 59{match_text};"):
        text = row['text']
        pid = row['PID']
        tid = row['TID']
        per_device_nvtx_rows[pid_to_device[pid]].append(row)
        assert pid in pid_to_device, f"PID {pid} not found in the pid to device map."
        event = {
                "name": text,
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "nvtx",
                "ts": munge_time(row["start"]),
                "dur": munge_time(row["end"] - row["start"]),
                "tid": "NVTX Thread {}".format(tid),
                "pid": "{} Device {}".format(filename, pid_to_device[pid]),
                "args": {
                    # TODO: More
                    },
                }
        if color_scheme:
            for key, color in color_scheme.items():
                if re.search(key, text):
                    event["cname"] = color
                    break
        per_device_nvtx_events[pid_to_device[pid]].append(event)
    return per_device_nvtx_rows, per_device_nvtx_events

def parse_cuda_api_events(conn: sqlite3.Connection, strings: dict, filename=""):
    """
    Copied from the docs:
    CUPTI_ACTIVITY_KIND_RUNTIME
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- CUDA event class enum value. See docs for specifics.
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL   REFERENCES StringIds(id), -- StringId of the function name.
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER   REFERENCES CUDA_CALLCHAINS(id) -- ID of the attached callchain.
    """
    pid_to_devices = link_pid_with_devices(conn)
    per_device_api_rows = defaultdict(list)
    per_device_api_events = defaultdict(list)
    # event type 0 is TRACE_PROCESS_EVENT_CUDA_RUNTIME
    for row in conn.execute(f"SELECT start, end, globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME;"):
        text = strings[row['nameId']]
        pid = row['PID']
        tid = row['TID']
        correlationId = row['correlationId']
        per_device_api_rows[pid_to_devices[pid]].append(row)
        event = {
                "name": text,
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda_api",
                "ts": munge_time(row["start"]),
                "dur": munge_time(row["end"] - row["start"]),
                "tid": "CUDA API Thread {}".format(tid),
                "pid": "{} Device {}".format(filename, pid_to_devices[pid]),
                "args": {
                        "correlationId": correlationId,
                    },
                }
        per_device_api_events[pid_to_devices[pid]].append(event)
    return per_device_api_rows, per_device_api_events

def _find_overlapping_intervals(nvtx_rows, cuda_api_rows):
    mixed_rows = []
    for nvtx_row in nvtx_rows:
        start = nvtx_row["start"]
        end = nvtx_row["end"]
        mixed_rows.append((start, 1, "nvtx", nvtx_row))
        mixed_rows.append((end, -1, "nvtx", nvtx_row))
    for cuda_api_row in cuda_api_rows:
        start = cuda_api_row["start"]
        end = cuda_api_row["end"]
        mixed_rows.append((start, 1, "cuda_api", cuda_api_row))
        mixed_rows.append((end, -1, "cuda_api", cuda_api_row))
    mixed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    active_intervals = []
    result = defaultdict(list)
    for _, event_type, event_origin, orig_event in mixed_rows:
        if event_type == 1:
            # start
            if event_origin == "nvtx":
                active_intervals.append(orig_event)
            else:
                for event in active_intervals:
                    result[event].append(orig_event)
        else:
            # end
            if event_origin == "nvtx":
                active_intervals.remove(orig_event)
    return result

def link_nvtx_events_to_kernel_events(strings: dict,
                                      pid_to_device: dict[int, int],
                                      per_device_nvtx_rows: dict[int, list],
                                      per_device_cuda_api_rows: dict[int, list],
                                      per_device_cuda_kernel_rows: dict[int, list],
                                      per_device_kernel_events: dict[int, list]):
    """
    Link NVTX events to cupti kernel events. This is done by first matching
    the nvtx ranges with CUDA API calls by timestamp. Then, retrieve the
    corresponding kernel events using the correlationId from CUDA API calls.
    """
    result = {}
    for device in pid_to_device.values():
        event_map = _find_overlapping_intervals(per_device_nvtx_rows[device], per_device_cuda_api_rows[device])
        correlation_id_map = defaultdict(dict)
        for cuda_api_row in per_device_cuda_api_rows[device]:
            correlation_id_map[cuda_api_row["correlationId"]]["cuda_api"] = cuda_api_row
        for kernel_row, kernel_trace_event in zip(per_device_cuda_kernel_rows[device], per_device_kernel_events[device]):
            correlation_id_map[kernel_row["correlationId"]]["kernel"] = kernel_row
            correlation_id_map[kernel_row["correlationId"]]["kernel_trace_event"] = kernel_trace_event
        for nvtx_row, cuda_api_rows in event_map.items():
            kernel_start_time = None
            kernel_end_time = None
            for cuda_api_row in cuda_api_rows:
                if "kernel" not in correlation_id_map[cuda_api_row["correlationId"]]:
                    # other cuda api event, ignore
                    continue
                kernel_row = correlation_id_map[cuda_api_row["correlationId"]]["kernel"]
                kernel_trace_event = correlation_id_map[cuda_api_row["correlationId"]]["kernel_trace_event"]
                if "NVTXRegions" not in kernel_trace_event["args"]:
                    kernel_trace_event["args"]["NVTXRegions"] = []
                kernel_trace_event["args"]["NVTXRegions"].append(nvtx_row["text"])
                if kernel_start_time is None or kernel_start_time > kernel_row["start"]:
                    kernel_start_time = kernel_row["start"]
                if kernel_end_time is None or kernel_end_time < kernel_row["end"]:
                    kernel_end_time = kernel_row["end"]
            if kernel_start_time is not None and kernel_end_time is not None:
                result[nvtx_row] = (kernel_start_time, kernel_end_time)
    return result

def parse_all_events(conn: sqlite3.Connection, strings: dict, activities=None, event_prefix=None, color_scheme={}, filename=""):
    if activities is None:
        activities = [ActivityType.KERNEL, ActivityType.NVTX_CPU, ActivityType.NVTX_KERNEL]
    if ActivityType.KERNEL in activities or ActivityType.NVTX_KERNEL in activities:
        per_device_kernel_rows, per_device_kernel_events = parse_cupti_kernel_events(conn, strings, filename=filename)
    if ActivityType.NVTX_CPU in activities or ActivityType.NVTX_KERNEL in activities:
        per_device_nvtx_rows, per_device_nvtx_events = parse_nvtx_events(conn, event_prefix=event_prefix, color_scheme=color_scheme, filename=filename)
    if ActivityType.CUDA_API in activities or ActivityType.NVTX_KERNEL in activities:
        per_device_cuda_api_rows, per_device_cuda_api_events = parse_cuda_api_events(conn, strings, filename=filename)
    if ActivityType.NVTX_KERNEL in activities:
        pid_to_device = link_pid_with_devices(conn)
        nvtx_kernel_event_map = link_nvtx_events_to_kernel_events(strings, pid_to_device, per_device_nvtx_rows, per_device_cuda_api_rows, per_device_kernel_rows, per_device_kernel_events)
    traceEvents = []
    if ActivityType.KERNEL in activities:
        for k, v in per_device_kernel_events.items():
            traceEvents.extend(v)
    if ActivityType.NVTX_CPU in activities:
        for k, v in per_device_nvtx_events.items():
            traceEvents.extend(v)
    if ActivityType.CUDA_API in activities:
        for k, v in per_device_cuda_api_events.items():
            traceEvents.extend(v)
    if ActivityType.NVTX_KERNEL in activities:
        for nvtx_event, (kernel_start_time, kernel_end_time) in nvtx_kernel_event_map.items():
            event = {
                "name": nvtx_event["text"] or "",
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "nvtx-kernel",
                "ts": munge_time(kernel_start_time),
                "dur": munge_time(kernel_end_time - kernel_start_time),
                "tid": "NVTX Kernel Thread {}".format(nvtx_event["tid"]),
                "pid": "{} Device {}".format(filename, pid_to_device[nvtx_event["pid"]]),
                "args": {
                    # TODO: More
                    },
                }
            traceEvents.append(event)
    return traceEvents

def align_traces(trace_events, based_on_event: str):
    # first separate out traces from different pids
    per_pid_traces = defaultdict(list)
    base_events_per_pid = defaultdict(list)
    for event in trace_events:
        pid = event["pid"]
        per_pid_traces[pid].append(event)
        if event["name"] == based_on_event:
            base_events_per_pid[pid].append(event)
    # now derive a time shift factor
    ref_pid = sorted(per_pid_traces.keys())[0]
    ref_events = base_events_per_pid[ref_pid]
    relative_time_shifts = defaultdict(list)
    for pid, base_events in base_events_per_pid.items():
        if pid == ref_pid:
            continue
        assert len(base_events) == len(ref_events), f"Number of base events ({len(base_events)}) does not match number of ref events ({len(ref_events)}) for pid {pid}"
        for i in range(len(base_events)):
            relative_time_shifts[pid].append(ref_events[i]["ts"] - base_events[i]["ts"])
            relative_time_shifts[pid].append((ref_events[i]["ts"] + ref_events[i]["dur"]) - (base_events[i]["ts"] + base_events[i]["dur"]))
    mean_time_shifts = {
        pid: sum(shifts) / len(shifts) for pid, shifts in relative_time_shifts.items()
    }
    # now apply the time shift factor
    for pid, events in per_pid_traces.items():
        if pid == ref_pid:
            continue
        for event in events:
            event["ts"] += mean_time_shifts[pid]
    # create new trace events
    new_trace_events = []
    for pid, events in per_pid_traces.items():
        new_trace_events.extend(events)
    new_trace_events.sort(key=lambda x: (x["pid"], x["tid"]))
    return new_trace_events


def nsys2json():
    args = parse_args()

    traceEvents = []
    for filename in args.filenames:
        conn = sqlite3.connect(filename)
        conn.row_factory = sqlite3.Row

        filename_base = os.path.splitext(os.path.basename(filename))[0]

        strings = {}
        for r in conn.execute("SELECT id, value FROM StringIds"):
            strings[r["id"]] = r["value"]

        events = parse_all_events(conn, strings, activities=args.activity_type, event_prefix=args.nvtx_event_prefix, color_scheme=args.nvtx_color_scheme, filename=filename_base)
        traceEvents.extend(events)
        conn.close()

    # make the timelines appear in pid and tid order
    traceEvents.sort(key=lambda x: (x["pid"], x["tid"]))

    if len(args.filenames) > 1 and args.align_traces:
        traceEvents = align_traces(traceEvents, args.align_traces)

    with open(args.output, 'w') as f:
        json.dump(traceEvents, f)

if __name__ == "__main__":
    nsys2json()
