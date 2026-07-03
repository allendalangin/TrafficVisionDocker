import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "trafficvision-dev-secret")

API_BASE_URL = os.environ.get("TRAFFICVISION_API_URL", "http://127.0.0.1:8000")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_api_json(path: str, *, params: Dict[str, Any] | None = None) -> Any:
    response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=60.0)
    response.raise_for_status()
    return response.json()


def delete_api_item(path: str) -> Any:
    response = httpx.delete(f"{API_BASE_URL}{path}", timeout=60.0)
    response.raise_for_status()
    return response.json()


def post_analysis(file_storage, confidence_threshold: float, batch_id: str | None = None) -> Dict[str, Any]:
    temp_path = UPLOAD_DIR / file_storage.filename
    file_storage.save(temp_path)

    try:
        with temp_path.open("rb") as image_file:
            params = {"confidence_threshold": confidence_threshold}
            if batch_id:
                params["batch_id"] = batch_id
            response = httpx.post(
                f"{API_BASE_URL}/analyze",
                files={"file": (file_storage.filename, image_file, file_storage.mimetype or "application/octet-stream")},
                params=params,
                timeout=120.0,
            )
        response.raise_for_status()
        return response.json()
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def post_batch_analysis(file_storage_list: List[Any], confidence_threshold: float) -> List[Dict[str, Any]]:
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    results: List[Dict[str, Any]] = []
    for file_storage in file_storage_list:
        result = post_analysis(file_storage, confidence_threshold, batch_id=batch_id)
        results.append(result)
    return results


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze() -> str:
    if request.method == "POST":
        image_file = request.files.get("image")
        if not image_file or image_file.filename == "":
            flash("Please choose an image to analyze.")
            return redirect(url_for("analyze"))

        confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
        try:
            result = post_analysis(image_file, confidence_threshold)
        except httpx.HTTPStatusError as exc:
            flash(f"Analysis request failed: {exc}")
            return redirect(url_for("analyze"))
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            flash(f"Unexpected error while analyzing image: {exc}")
            return redirect(url_for("analyze"))

        return render_template("result.html", result=result)

    return render_template("analyze.html")


@app.route("/batch-analyze", methods=["GET", "POST"])
def batch_analyze() -> str:
    if request.method == "POST":
        image_files = request.files.getlist("images")
        image_files = [f for f in image_files if f and f.filename]
        if not image_files:
            flash("Please choose at least one image to analyze.")
            return redirect(url_for("batch_analyze"))

        confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
        try:
            results = post_batch_analysis(image_files, confidence_threshold)
        except httpx.HTTPStatusError as exc:
            flash(f"Batch analysis request failed: {exc}")
            return redirect(url_for("batch_analyze"))
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            flash(f"Unexpected error while analyzing images: {exc}")
            return redirect(url_for("batch_analyze"))

        return render_template("batch_result.html", results=results)

    return render_template("batch_analyze.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory("uploads", filename)


@app.route("/history")
def history() -> str:
    try:
        analyses = get_api_json("/history")
    except httpx.HTTPError as exc:
        flash(f"Unable to load history: {exc}")
        analyses = []

    grouped_items = []
    batch_map: Dict[str, List[Dict[str, Any]]] = {}
    for item in analyses:
        batch_id = item.get("batchId")
        if batch_id:
            batch_map.setdefault(batch_id, []).append(item)
        else:
            grouped_items.append({"kind": "single", "item": item})

    for batch_id, batch_items in batch_map.items():
        grouped_items.append({"kind": "batch", "batch_id": batch_id, "items": batch_items})

    grouped_items.sort(
        key=lambda entry: (
            entry["item"]["timestamp"] if entry["kind"] == "single" else entry["items"][-1]["timestamp"]
        ),
        reverse=True,
    )

    return render_template("history.html", analyses=grouped_items)


@app.route("/history/batch/<batch_id>", methods=["GET", "POST"])
def batch_history_detail(batch_id: str) -> str:
    if request.method == "POST":
        try:
            analyses = get_api_json("/history")
            for item in analyses:
                if item.get("batchId") == batch_id:
                    delete_api_item(f"/analysis/{item['analysisId']}")
            flash("Batch deleted.")
        except httpx.HTTPError as exc:
            flash(f"Unable to delete batch: {exc}")
        return redirect(url_for("history"))

    try:
        analyses = get_api_json("/history")
    except httpx.HTTPError as exc:
        flash(f"Unable to load history: {exc}")
        return redirect(url_for("history"))

    batch_items = [item for item in analyses if item.get("batchId") == batch_id]
    if not batch_items:
        flash("That batch was not found.")
        return redirect(url_for("history"))

    return render_template("batch_detail.html", batch_id=batch_id, analyses=batch_items)


@app.route("/history/<analysis_id>", methods=["GET", "POST"])
def analysis_detail(analysis_id: str) -> str:
    if request.method == "POST":
        try:
            delete_api_item(f"/analysis/{analysis_id}")
            flash("Analysis deleted.")
        except httpx.HTTPError as exc:
            flash(f"Unable to delete analysis: {exc}")
        return redirect(url_for("history"))

    try:
        analysis = get_api_json(f"/analysis/{analysis_id}")
    except httpx.HTTPError as exc:
        flash(f"Unable to load analysis details: {exc}")
        return redirect(url_for("history"))

    return render_template("detail.html", analysis=analysis)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
