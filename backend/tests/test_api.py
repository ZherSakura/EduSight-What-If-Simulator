"""
EduSight backend API tests.

"""

VALID_STUDENT = "STU-0001"
UNKNOWN_STUDENT = "STU-9999"


# ----- Health ---------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "EduSight Backend API"


# ----- GET /students --------------------------------------------------------

def test_list_students_returns_data(client):
    resp = client.get("/students")
    assert resp.status_code == 200
    students = resp.json()
    assert isinstance(students, list)
    assert len(students) > 0


def test_list_students_respects_limit(client):
    resp = client.get("/students", params={"limit": 5})
    assert resp.status_code == 200
    assert len(resp.json()) <= 5


def test_list_students_has_expected_fields(client):
    resp = client.get("/students", params={"limit": 1})
    student = resp.json()[0]
    for field in ("id", "name", "form", "school", "attendance",
                  "grades", "currentScore", "currentRisk"):
        assert field in student, f"missing field: {field}"


def test_list_students_rejects_bad_limit(client):
    # limit above the allowed maximum (200) must be a validation error
    resp = client.get("/students", params={"limit": 999})
    assert resp.status_code == 422


# ----- GET /students/id -----------------------------------------------------

def test_get_single_student(client):
    resp = client.get(f"/students/{VALID_STUDENT}")
    assert resp.status_code == 200
    student = resp.json()
    assert student["id"] == VALID_STUDENT
    assert 0 <= student["currentScore"] <= 100


def test_get_unknown_student_returns_404(client):
    resp = client.get(f"/students/{UNKNOWN_STUDENT}")
    assert resp.status_code == 404


# ----- POST /simulate -------------------------------------------------------

def _simulate_body(student_id=VALID_STUDENT, **overrides):
    inputs = {"attendance": 90, "grades": 75, "counselling": 2, "welfare": 1}
    inputs.update(overrides)
    return {"studentId": student_id, "inputs": inputs}


def test_simulate_happy_path(client):
    resp = client.post("/simulate", json=_simulate_body())
    assert resp.status_code == 200
    result = resp.json()

    assert 0 <= result["baselineScore"] <= 100
    assert 0 <= result["projectedScore"] <= 100
    assert result["riskLevel"] in ("low", "medium", "high")
    assert 0 <= result["dropoutProbability"] <= 100
    assert isinstance(result["recommendations"], list)
    assert result["narrative"]

    weights = result["weights"]
    for factor in ("attendance", "academic", "socioeconomic", "family"):
        assert factor in weights


def test_simulate_improvement_lowers_score(client):
    resp = client.post("/simulate", json=_simulate_body(attendance=100, grades=100,
                                                        counselling=10, welfare=2))
    result = resp.json()
    assert result["projectedScore"] <= result["baselineScore"]


def test_simulate_unknown_student_returns_404(client):
    resp = client.post("/simulate", json=_simulate_body(student_id=UNKNOWN_STUDENT))
    assert resp.status_code == 404


def test_simulate_rejects_out_of_range_attendance(client):
    resp = client.post("/simulate", json=_simulate_body(attendance=150))
    assert resp.status_code == 422


def test_simulate_rejects_negative_counselling(client):
    resp = client.post("/simulate", json=_simulate_body(counselling=-1))
    assert resp.status_code == 422


def test_simulate_rejects_missing_body(client):
    resp = client.post("/simulate", json={})
    assert resp.status_code == 422


def test_simulate_includes_ml_fields(client):
    result = client.post("/simulate", json=_simulate_body()).json()
    assert "mlScore" in result
    assert result["mlSource"] in ("cloud", "local", "none")
