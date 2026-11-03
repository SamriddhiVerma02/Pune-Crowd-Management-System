import statistics
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_video(video_path):
    """Estimate the number of distinct people visible anywhere in the video."""
    video_file = Path(video_path)

    if not video_file.exists() or not video_file.is_file():
        return {
            "peopleCount": 0,
            "totalFramesProcessed": 0,
            "averagePeopleCount": 0.0,
            "maximumPeopleCount": 0,
            "error": f"Video not found or unreadable: {video_file}",
        }

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return {
            "peopleCount": 0,
            "totalFramesProcessed": 0,
            "averagePeopleCount": 0.0,
            "maximumPeopleCount": 0,
            "error": f"Could not open video: {video_file}",
        }

    model_dir = Path(__file__).resolve().parent
    model_candidates = [model_dir / "yolov8s.pt", model_dir / "yolov8n.pt"]
    model_path = next((str(path) for path in model_candidates if path.exists()), "yolov8s.pt")
    model = YOLO(model_path)

    debug_dir = video_file.parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    debug_frame_path = debug_dir / f"{video_file.stem}_tracked_debug.png"

    frame_counts = []
    processed_frames = 0
    sample_interval = 5
    conf_threshold = 0.35
    imgsz = 1280
    debug_saved = False
    people_registry = []
    active_track_lookup = {}
    next_person_id = 1

    def box_iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def compute_appearance_feature(frame, box):
        x1, y1, x2, y2 = [
            int(max(0, min(frame.shape[1], coord)))
            for coord in [box[0], box[1], box[2], box[3]]
        ]
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if crop.size == 0:
            return np.zeros(32 * 32, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def find_matching_person(detection, frame_shape):
        best_match = None
        best_score = -1.0

        for person in people_registry:
            if person["last_seen_frame"] < processed_frames - 25:
                continue

            last_box = person.get("last_box")
            if last_box is None:
                continue

            iou = box_iou(last_box, detection["box"])
            center_dist = np.linalg.norm(np.array(detection["center"]) - np.array(person["center"]))
            frame_diag = float(np.hypot(frame_shape[1], frame_shape[0]))
            appearance_similarity = float(np.dot(person["appearance"], detection["appearance"]))

            spatial_ok = iou > 0.08 or center_dist <= max(45.0, frame_diag * 0.12)
            appearance_ok = appearance_similarity > 0.50
            if not (spatial_ok and appearance_ok):
                continue

            score = appearance_similarity + 0.8 * iou
            if score > best_score:
                best_score = score
                best_match = person

        return best_match

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            break

        processed_frames += 1
        if processed_frames % sample_interval != 0:
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=conf_threshold,
            imgsz=imgsz,
            classes=[0],
            verbose=False,
            iou=0.5,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            frame_counts.append(0)
            continue

        tracked_ids = []
        box_ids = getattr(boxes, "id", None)
        if box_ids is not None:
            try:
                tracked_ids = [int(track_id) for track_id in box_ids.cpu().tolist()]
            except Exception:
                tracked_ids = []

        detections = []
        for idx, box_data in enumerate(boxes):
            track_id = tracked_ids[idx] if idx < len(tracked_ids) else None
            if hasattr(box_data, "xyxy"):
                xyxy = box_data.xyxy.cpu().numpy()[0]
            else:
                xyxy = box_data
            x1, y1, x2, y2 = [float(coord) for coord in xyxy]
            box = [x1, y1, x2, y2]
            detections.append({
                "track_id": track_id,
                "box": box,
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "appearance": compute_appearance_feature(frame, box),
            })

        frame_people = []
        frame_identity_map = {}

        for detection in detections:
            track_id = detection["track_id"]

            if track_id is not None and track_id in active_track_lookup:
                person = active_track_lookup[track_id]
                person["last_box"] = detection["box"]
                person["center"] = detection["center"]
                person["appearance"] = detection["appearance"]
                person["last_seen_frame"] = processed_frames
                if track_id not in person["track_ids"]:
                    person["track_ids"].add(track_id)
                frame_people.append(person["person_id"])
                frame_identity_map[track_id] = person["person_id"]
                continue

            match = find_matching_person(detection, frame.shape)
            if match is not None:
                match["last_box"] = detection["box"]
                match["center"] = detection["center"]
                match["appearance"] = detection["appearance"]
                match["last_seen_frame"] = processed_frames
                if track_id is not None:
                    match["track_ids"].add(track_id)
                    active_track_lookup[track_id] = match
                frame_people.append(match["person_id"])
                frame_identity_map[track_id if track_id is not None else id(detection)] = match["person_id"]
                continue

            new_person = {
                "person_id": next_person_id,
                "track_ids": set(),
                "last_box": detection["box"],
                "center": detection["center"],
                "appearance": detection["appearance"],
                "last_seen_frame": processed_frames,
            }
            if track_id is not None:
                new_person["track_ids"].add(track_id)
                active_track_lookup[track_id] = new_person
            people_registry.append(new_person)
            next_person_id += 1
            frame_people.append(new_person["person_id"])
            frame_identity_map[track_id if track_id is not None else id(detection)] = new_person["person_id"]

        current_count = len(set(frame_people)) if frame_people else 0
        frame_counts.append(current_count)

        if not debug_saved and detections:
            try:
                debug_frame = frame.copy()
                for detection in detections:
                    detection_id = detection["track_id"] if detection["track_id"] is not None else None
                    person_id = None
                    if detection_id is not None and detection_id in active_track_lookup:
                        person_id = active_track_lookup[detection_id]["person_id"]
                    elif detection_id is not None:
                        person_id = frame_identity_map.get(detection_id)
                    else:
                        person_id = frame_identity_map.get(id(detection))

                    if person_id is None:
                        person_id = "?"

                    x1, y1, x2, y2 = [int(coord) for coord in detection["box"]]
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        debug_frame,
                        f"id:{person_id}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                cv2.imwrite(str(debug_frame_path), debug_frame)
                debug_saved = True
            except Exception:
                pass

        active_track_lookup = {
            track_id: person for track_id, person in active_track_lookup.items() if person["last_seen_frame"] >= processed_frames - 20
        }

    cap.release()

    if not frame_counts:
        return {
            "peopleCount": 0,
            "totalFramesProcessed": 0,
            "averagePeopleCount": 0.0,
            "maximumPeopleCount": 0,
            "debugFramePath": str(debug_frame_path) if debug_frame_path.exists() else None,
        }

    average_count = statistics.mean(frame_counts)
    max_count = max(frame_counts)

    return {
        "peopleCount": len(people_registry),
        "totalFramesProcessed": len(frame_counts),
        "averagePeopleCount": round(average_count, 2),
        "maximumPeopleCount": max_count,
        "debugFramePath": str(debug_frame_path) if debug_frame_path.exists() else None,
    }


@app.get("/api/test")
def test():
    return {"message": "Backend is working!"}

def convert_area_to_square_metres(area, area_unit):
    if area is None or area_unit is None:
        return None

    normalized = str(area_unit).strip().lower()

    if normalized in {"m2", "sqm", "sq m", "square metre", "square meters", "square meter"}:
        return float(area)
    if normalized in {"acre", "acres"}:
        return float(area) * 4046.8564224

    return None


def calculate_estimated_capacity(place):
    area = place.get("area")
    area_unit = place.get("areaUnit")
    usable_area_percentage = place.get("usableAreaPercentage")
    density_limit = place.get("densityLimit")

    if area is not None and area_unit is not None and usable_area_percentage is not None and density_limit is not None:
        area_sq_m = convert_area_to_square_metres(area, area_unit)
        if area_sq_m is not None:
            usable_area = area_sq_m * usable_area_percentage
            estimated_capacity = usable_area * density_limit
            return max(0, int(round(estimated_capacity)))

    fallback_area_sq_m = place.get("fallbackAreaSqM")
    fallback_usable_area_percentage = place.get("fallbackUsableAreaPercentage")
    fallback_density_limit = place.get("fallbackDensityLimit")

    if fallback_area_sq_m is not None and fallback_usable_area_percentage is not None and fallback_density_limit is not None:
        usable_area = fallback_area_sq_m * fallback_usable_area_percentage
        estimated_capacity = usable_area * fallback_density_limit
        return max(0, int(round(estimated_capacity)))

    return None


PLACE_FALLBACKS = {
    "Shaniwar Wada": {
        "fallbackAreaSqM": 12000,
        "fallbackUsableAreaPercentage": 0.20,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative heritage-site area proxy and tourist-density assumption; not an official limit.",
    },
    "Aga Khan Palace": {
        "fallbackAreaSqM": None,
        "fallbackUsableAreaPercentage": None,
        "fallbackDensityLimit": None,
        "capacityNote": "Estimated from documented site area and a conservative usable-area fraction; not an official limit.",
    },
    "Dagdusheth Halwai Ganpati Temple": {
        "fallbackAreaSqM": 3200,
        "fallbackUsableAreaPercentage": 0.25,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative temple-floor proxy and tourist-density assumption; not an official limit.",
    },
    "Sinhagad Fort": {
        "fallbackAreaSqM": 24000,
        "fallbackUsableAreaPercentage": 0.20,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative fort-area proxy and tourist-density assumption; not an official limit.",
    },
    "Parvati Hill": {
        "fallbackAreaSqM": 18000,
        "fallbackUsableAreaPercentage": 0.18,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative hill-site area proxy and tourist-density assumption; not an official limit.",
    },
    "Pataleshwar Caves": {
        "fallbackAreaSqM": 1800,
        "fallbackUsableAreaPercentage": 0.25,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative cave-area proxy and tourist-density assumption; not an official limit.",
    },
    "Raja Dinkar Kelkar Museum": {
        "fallbackAreaSqM": 7000,
        "fallbackUsableAreaPercentage": 0.28,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative museum-floor proxy and tourist-density assumption; not an official limit.",
    },
    "Okayama Friendship Garden": {
        "fallbackAreaSqM": 16000,
        "fallbackUsableAreaPercentage": 0.25,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative garden-area proxy and tourist-density assumption; not an official limit.",
    },
    "Vishrambaug Wada": {
        "fallbackAreaSqM": 5000,
        "fallbackUsableAreaPercentage": 0.25,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative heritage-site area proxy and tourist-density assumption; not an official limit.",
    },
    "Khadakwasla Dam": {
        "fallbackAreaSqM": 30000,
        "fallbackUsableAreaPercentage": 0.18,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative recreation-area proxy and tourist-density assumption; not an official limit.",
    },
    "Rajiv Gandhi Zoological Park": {
        "fallbackAreaSqM": None,
        "fallbackUsableAreaPercentage": None,
        "fallbackDensityLimit": None,
        "capacityNote": "Estimated from documented land area and a conservative usable-area fraction; not an official visitor limit.",
    },
    "Darshan Museum": {
        "fallbackAreaSqM": 4000,
        "fallbackUsableAreaPercentage": 0.30,
        "fallbackDensityLimit": 0.12,
        "capacityNote": "Estimated from a conservative museum-floor proxy and tourist-density assumption; not an official limit.",
    },
}


@app.get("/api/places")
def get_places():
    places = [
        {
            "id": 1,
            "name": "Shaniwar Wada",
            "description": "A historic fort and one of Pune's most famous landmarks.",
            "address": "Shaniwar Wada, Kasba Peth, Pune",
            "openingHours": "9:00 AM - 6:00 PM",
            "rating": 4.6,
            "reviewCount": 1280,
            "entryFee": "₹25",
            "peopleCount": 180,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Shaniwar Wada"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Shaniwar Wada"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative heritage-site area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Shaniwar_Wada",
            "capacityNote": PLACE_FALLBACKS["Shaniwar Wada"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Shaniwar Wada"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Shaniwar Wada"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Shaniwar Wada"]["fallbackDensityLimit"],
        },
        {
            "id": 2,
            "name": "Aga Khan Palace",
            "description": "A historic palace surrounded by peaceful gardens.",
            "address": "Aga Khan Palace, Kalyani Nagar, Pune",
            "openingHours": "9:30 AM - 5:30 PM",
            "rating": 4.7,
            "reviewCount": 950,
            "entryFee": "₹20",
            "peopleCount": 140,
            "area": 19,
            "areaUnit": "acres",
            "usableAreaPercentage": 0.06,
            "densityLimit": 0.12,
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Documented site-area estimate using public site area; not an official visitor limit.",
            "capacitySourceUrl": "https://www.agakhanpalacepune.org/",
            "capacityNote": PLACE_FALLBACKS["Aga Khan Palace"]["capacityNote"],
            "capacityAvailable": True,
            "fallbackAreaSqM": None,
            "fallbackUsableAreaPercentage": None,
            "fallbackDensityLimit": None,
        },
        {
            "id": 3,
            "name": "Dagdusheth Halwai Ganpati Temple",
            "description": "A famous Ganpati temple located in the heart of Pune.",
            "address": "Dagdusheth Halwai Ganpati Temple, Budhwar Peth, Pune",
            "openingHours": "5:00 AM - 10:00 PM",
            "rating": 4.8,
            "reviewCount": 2200,
            "entryFee": "Free",
            "peopleCount": 260,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative temple-floor proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": None,
            "capacityNote": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Dagdusheth Halwai Ganpati Temple"]["fallbackDensityLimit"],
        },
        {
            "id": 4,
            "name": "Sinhagad Fort",
            "description": "A historic hill fort offering beautiful views of the surrounding region.",
            "address": "Sinhagad Road, Pune",
            "openingHours": "6:00 AM - 6:00 PM",
            "rating": 4.5,
            "reviewCount": 1750,
            "entryFee": "₹50",
            "peopleCount": 220,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Sinhagad Fort"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Sinhagad Fort"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative fort-area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Sinhagad_Fort",
            "capacityNote": PLACE_FALLBACKS["Sinhagad Fort"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Sinhagad Fort"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Sinhagad Fort"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Sinhagad Fort"]["fallbackDensityLimit"],
        },
        {
            "id": 5,
            "name": "Parvati Hill",
            "description": "A historic hilltop temple complex with panoramic views of Pune.",
            "address": "Parvati Hill, Parvati, Pune",
            "openingHours": "5:00 AM - 8:00 PM",
            "rating": 4.4,
            "reviewCount": 870,
            "entryFee": "Free",
            "peopleCount": 160,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Parvati Hill"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Parvati Hill"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative hill-site area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Parvati_Hill",
            "capacityNote": PLACE_FALLBACKS["Parvati Hill"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Parvati Hill"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Parvati Hill"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Parvati Hill"]["fallbackDensityLimit"],
        },
        {
            "id": 6,
            "name": "Pataleshwar Caves",
            "description": "An ancient rock-cut cave temple in central Pune.",
            "address": "Pataleshwar Road, Shivaji Nagar, Pune",
            "openingHours": "8:00 AM - 5:30 PM",
            "rating": 4.3,
            "reviewCount": 660,
            "entryFee": "₹10",
            "peopleCount": 110,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Pataleshwar Caves"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Pataleshwar Caves"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative cave-area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Pataleshwar_Caves",
            "capacityNote": PLACE_FALLBACKS["Pataleshwar Caves"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Pataleshwar Caves"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Pataleshwar Caves"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Pataleshwar Caves"]["fallbackDensityLimit"],
        },
        {
            "id": 7,
            "name": "Raja Dinkar Kelkar Museum",
            "description": "A museum showcasing traditional Indian art and artifacts.",
            "address": "Shukrawar Peth, Pune",
            "openingHours": "10:00 AM - 5:00 PM",
            "rating": 4.6,
            "reviewCount": 900,
            "entryFee": "₹50",
            "peopleCount": 130,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative museum-floor proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://www.kelkarmuseum.org/",
            "capacityNote": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Raja Dinkar Kelkar Museum"]["fallbackDensityLimit"],
        },
        {
            "id": 8,
            "name": "Okayama Friendship Garden",
            "description": "A Japanese-style garden offering a peaceful place to relax.",
            "address": "Okayama Friendship Garden, Baner, Pune",
            "openingHours": "9:00 AM - 8:00 PM",
            "rating": 4.5,
            "reviewCount": 510,
            "entryFee": "₹30",
            "peopleCount": 90,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Okayama Friendship Garden"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Okayama Friendship Garden"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative garden-area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Okayama_Friendship_Garden",
            "capacityNote": PLACE_FALLBACKS["Okayama Friendship Garden"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Okayama Friendship Garden"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Okayama Friendship Garden"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Okayama Friendship Garden"]["fallbackDensityLimit"],
        },
        {
            "id": 9,
            "name": "Vishrambaug Wada",
            "description": "A historic mansion representing Pune's Maratha-era architecture.",
            "address": "Vishrambaug Wada, Sadashiv Peth, Pune",
            "openingHours": "9:00 AM - 6:00 PM",
            "rating": 4.2,
            "reviewCount": 430,
            "entryFee": "₹25",
            "peopleCount": 80,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Vishrambaug Wada"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Vishrambaug Wada"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative heritage-site area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Vishrambaug_Wada",
            "capacityNote": PLACE_FALLBACKS["Vishrambaug Wada"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Vishrambaug Wada"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Vishrambaug Wada"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Vishrambaug Wada"]["fallbackDensityLimit"],
        },
        {
            "id": 10,
            "name": "Khadakwasla Dam",
            "description": "A popular scenic destination near Pune known for its reservoir.",
            "address": "Khadakwasla, Pune",
            "openingHours": "6:00 AM - 8:00 PM",
            "rating": 4.4,
            "reviewCount": 740,
            "entryFee": "Free",
            "peopleCount": 210,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Khadakwasla Dam"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Khadakwasla Dam"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative recreation-area proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Khadakwasla_Dam",
            "capacityNote": PLACE_FALLBACKS["Khadakwasla Dam"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Khadakwasla Dam"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Khadakwasla Dam"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Khadakwasla Dam"]["fallbackDensityLimit"],
        },
        {
            "id": 11,
            "name": "Rajiv Gandhi Zoological Park",
            "description": "A popular zoo and wildlife park in Pune.",
            "address": "Rajiv Gandhi Zoological Park, Katraj, Pune",
            "openingHours": "9:00 AM - 6:00 PM",
            "rating": 4.3,
            "reviewCount": 1100,
            "entryFee": "₹80",
            "peopleCount": 270,
            "area": 130,
            "areaUnit": "acres",
            "usableAreaPercentage": 0.04,
            "densityLimit": 0.12,
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Documented land-area estimate using public zoo area; not an official visitor limit.",
            "capacitySourceUrl": "https://en.wikipedia.org/wiki/Rajiv_Gandhi_Zoological_Park",
            "capacityNote": PLACE_FALLBACKS["Rajiv Gandhi Zoological Park"]["capacityNote"],
            "capacityAvailable": True,
            "fallbackAreaSqM": None,
            "fallbackUsableAreaPercentage": None,
            "fallbackDensityLimit": None,
        },
        {
            "id": 12,
            "name": "Darshan Museum",
            "description": "A museum dedicated to the life and teachings of Sadhu Vaswani.",
            "address": "Darshan Museum, Pune",
            "openingHours": "10:00 AM - 6:00 PM",
            "rating": 4.6,
            "reviewCount": 610,
            "entryFee": "₹60",
            "peopleCount": 95,
            "area": None,
            "areaUnit": None,
            "usableAreaPercentage": PLACE_FALLBACKS["Darshan Museum"]["fallbackUsableAreaPercentage"],
            "densityLimit": PLACE_FALLBACKS["Darshan Museum"]["fallbackDensityLimit"],
            "capacity": None,
            "estimatedCapacity": None,
            "capacitySource": "Estimated from a conservative museum-floor proxy and tourist-density assumption; not an official limit.",
            "capacitySourceUrl": "https://darshanmuseum.org/",
            "capacityNote": PLACE_FALLBACKS["Darshan Museum"]["capacityNote"],
            "capacityAvailable": False,
            "fallbackAreaSqM": PLACE_FALLBACKS["Darshan Museum"]["fallbackAreaSqM"],
            "fallbackUsableAreaPercentage": PLACE_FALLBACKS["Darshan Museum"]["fallbackUsableAreaPercentage"],
            "fallbackDensityLimit": PLACE_FALLBACKS["Darshan Museum"]["fallbackDensityLimit"],
        },
    ]

    for place in places:
        estimated_capacity = calculate_estimated_capacity(place)
        if estimated_capacity is None:
            fallback_area = place.get("fallbackAreaSqM")
            fallback_usable = place.get("fallbackUsableAreaPercentage")
            fallback_density = place.get("fallbackDensityLimit")
            if fallback_area is not None and fallback_usable is not None and fallback_density is not None:
                estimated_capacity = int(round(fallback_area * fallback_usable * fallback_density))
        place["capacity"] = estimated_capacity if estimated_capacity is not None else 0
        place["estimatedCapacity"] = place["capacity"]
        place["capacityAvailable"] = True

    return places
@app.post("/api/crowd/calculate")
def calculate_crowd(peopleCount: int, capacity: int):

    if capacity is None or capacity <= 0:
        return {
            "error": "Capacity is unavailable or must be greater than 0",
            "capacityAvailable": False
        }

    crowdScore = round((peopleCount / capacity) * 100)

    if crowdScore <= 30:
        crowdLevel = "CALM"
    elif crowdScore <= 55:
        crowdLevel = "STEADY"
    elif crowdScore <= 78:
        crowdLevel = "BUSY"
    else:
        crowdLevel = "PACKED"

    return {
        "peopleCount": peopleCount,
        "capacity": capacity,
        "crowdScore": crowdScore,
        "crowdLevel": crowdLevel
    }


@app.post("/api/crowd/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        return {
            "success": False,
            "filename": "",
            "message": "No file selected."
        }

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "filename": file.filename,
            "message": "Unsupported file type. Please upload mp4, mov, avi, or webm."
        }

    upload_path = UPLOAD_FOLDER / file.filename

    # Save the uploaded video to the existing uploads folder.
    with upload_path.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    # Analyze only the newly saved video file.
    try:
        analysis = analyze_video(str(upload_path))
    except Exception as exc:
        return {
            "success": False,
            "filename": file.filename,
            "message": f"Video uploaded, but analysis failed: {str(exc)}",
        }

    return {
        "success": True,
        "filename": file.filename,
        "message": "Video uploaded successfully.",
        "analysis": {
            "peopleCount": analysis.get("peopleCount", 0),
            "totalFramesProcessed": analysis.get("totalFramesProcessed", 0),
            "averagePeopleCount": analysis.get("averagePeopleCount", 0.0),
            "maximumPeopleCount": analysis.get("maximumPeopleCount", 0),
        },
    }