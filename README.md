# Meridian Hospital Network — MCP Server Lab

**Company:** Meridian Hospital Network
*(fictional two-hospital regional group: MediCore Downtown, MediCore North)*

---

## Problem

Front-desk and hospital staff currently rely on manual communication to check available resources such as ICU beds and operating rooms, then update patient admission information manually.

During busy periods, staff may use general-purpose AI assistants to retrieve hospital information or perform database updates.

Allowing an AI model to directly interact with a hospital database without restrictions creates serious risks:

- Assigning an unavailable ICU bed.
- Updating patient information incorrectly.
- Modifying hospital resources without authorization.
- Performing unsafe write operations.

To prevent these failures, we implemented an **MCP server** between the AI agent and the hospital database.

The model can retrieve information through controlled read-only tools, while every write operation is performed through typed MCP tools with:

- Server-side validation.
- JSON Schema validation.
- Authorization checks.
- Business rule enforcement.

This README documents **Member 3's contribution**, including MCP client integration, MCP protocol features, testing, and demonstration.

---

## Repository Layout

```text
.
├── .gitignore
├── README.md
├── requirements.txt
│
├── agent/
│   ├── agent.py
│   ├── mcp_protocol.py
│   └── test_e2e.py
│
├── db/
│   ├── drawsql_erd.png
│   ├── erd_dbdiagram.png
│   ├── README.md
│   ├── schema.sql
│   └── seed.sql
│
└── mcp_server/
    ├── MCP.py
    ├── db_helpers.py
    ├── mock_server.py
    ├── schemas.py
    └── validation.py
```

---

## MCP Server Components

### MCP.py

Main MCP server implementation.

Responsibilities:

- Define MCP tools.
- Handle database operations.
- Apply validation rules.
- Expose MCP resources and prompts.

### schemas.py

Contains JSON Schemas used for validating MCP tool inputs.

Includes schemas for:

- Patient registration.
- Patient status updates.
- Admissions.
- ICU bed assignments.
- Operating room status updates.

### validation.py

Responsible for security and business rule validation.

Includes:

- Authorization checks.
- Patient validation.
- ICU assignment validation.
- Admission validation.
- Operating room validation.

### db_helpers.py

Database helper functions responsible for communication with the hospital database layer.

### mock_server.py

Mock MCP server used for end-to-end testing without requiring a real database connection.

---

## Running the Project

```bash
cd agent

pip install -r ../requirements.txt

python test_e2e.py

python agent.py --demo

python agent.py
```

If `ANTHROPIC_API_KEY` is not configured, the client automatically uses an offline deterministic planner and sampling stub.

If the API key is available, Claude is used for model-based tool selection and sampling.

---

## MCP Tools

| Tool | Type | Authentication | Human Confirmation | Purpose |
|---|---|---|---|---|
| `get_patient_details` | Read | None | No | Retrieve complete patient information by ID |
| `get_available_icu_beds` | Read | None | No | Retrieve available ICU beds |
| `get_hospital_capacity` | Read | None | No | Check hospital capacity information |
| `register_patient` | Write | Admin | No | Register a new patient |
| `update_patient_status` | Write | Doctor | Policy dependent | Update patient medical status |
| `create_admission` | Write | Doctor | No | Create admission record |
| `manage_icu_bed` | Write | Doctor | Yes for sensitive assignments | Assign or release ICU beds |
| `update_operating_room_status` | Write | Admin | No | Update operating room status |

---

## Capability Negotiation

During MCP initialization, the client exchanges capabilities with the server.

Supported capabilities:

- Tools
- Resources
- Prompts
- Elicitation
- Sampling
- Progress Notifications

The client stores server capabilities and checks availability before using optional MCP features.

Example:

```python
agent.supports("elicitation")
agent.supports("sampling")
```

---

## MCP Features Implemented

### Capability Negotiation

The client exchanges capabilities during the MCP initialize phase.

The server responds with supported features and the client adapts its behavior accordingly.

### Notifications

The server supports progress notifications for long-running operations.

Example:

```
notifications/progress
```

Used to provide updates while checking hospital resources.

Example:

```
Checking ICU beds...
```

### Elicitation (Human-in-the-Loop)

Sensitive operations require explicit user confirmation before execution.

Example:

```
Confirm ICU bed assignment?
```

Implemented for:

- ICU bed assignment using `manage_icu_bed`.
- Critical resource allocation operations.

This prevents unsafe automatic modifications.

### Sampling

The server can request the client model to generate content.

Example:

```
sampling/createMessage
```

Used for AI-generated admission-related text.

If no API key exists, an offline sampling stub is used.

### Resources

Hospital policies are exposed as MCP Resources instead of tools.

Available resources:

- `triage://protocols/guidelines`

  Emergency triage guidelines.

- `hospital://operating-rooms/rules`

  Operating room rules and policies.

### Prompts

Parameterized prompts are available through MCP prompts.

Available prompt:

- `triage_patient_prompt`

  Purpose:

  - Analyze patient urgency.
  - Use hospital guidelines.
  - Select suitable MCP tools.

### Defensive Tool Design

The MCP server applies multiple protection layers.

**Input Validation**

Implemented using:

- Pydantic models.
- JSON Schema validation.

**Authorization**

Write operations require proper authorization.

Examples:

- Doctors can update patient medical information.
- Admin users can register patients and manage operating room status.

**Business Rules**

The server validates:

- Patient IDs.
- Allowed patient statuses.
- ICU bed assignments.
- Admission data.
- Operating room states.

### Transport

The MCP client communicates with the mock server through:

- stdio

This follows the MCP local development workflow.

---

## Demo

Server capabilities:

```json
{
  "tools": {
    "listChanged": true
  },
  "elicitation": {},
  "sampling": {},
  "progress": {}
}
```

Available Tools:

```
get_patient_details
get_available_icu_beds
get_hospital_capacity
register_patient
update_patient_status
create_admission
manage_icu_bed
update_operating_room_status
```

**USER:**
Register a new patient

**Result:**
Patient registered successfully.

**USER:**
Get patient details

**Result:**
Patient details retrieved successfully.

**USER:**
Check available ICU beds

**Result:**
Available ICU beds returned successfully.

**USER:**
Assign ICU bed to patient

Human confirmation requested:

```
Confirm ICU bed assignment?
```

**Result:**
ICU bed assigned successfully.

**USER:**
Create admission

**Result:**
Admission created successfully.

Sampling request completed when supported.

---

## End-to-End Tests

Run:

```bash
python test_e2e.py
```

Expected output:

```
PASS: Capability Negotiation
PASS: Notifications
PASS: Human Confirmation
PASS: Resources
PASS: Prompts
PASS: Progress Tracking
PASS: Defensive Tool Design
PASS: Sampling

All tests passed
```

---

## Production Considerations

Although this prototype demonstrates MCP protocol concepts, production deployment would require:

- Secure authentication (JWT or similar).
- Complete audit logging for every write operation.
- Support for multiple simultaneous confirmation requests.
- Production-grade model integration.
- Connection with a real hospital database.
- Additional security monitoring.
