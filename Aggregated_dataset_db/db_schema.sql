-- SQLite schema for occlusion detection database.
-- Derived from db_schema_review.md.

CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    venue_type TEXT NOT NULL CHECK (venue_type IN ('porta','corridoio','scala')),
    is_normal BOOLEAN NOT NULL,
    obstacle_class TEXT,
    occlusion_type TEXT NOT NULL CHECK (
        occlusion_type IN ('none','synthetic_geometric','synthetic_copypaste','real')
    ),
    occlusion_level TEXT NOT NULL CHECK (
        occlusion_level IN ('none','partial','full')
    ),
    split TEXT NOT NULL CHECK (split IN ('train','val','test')),
    source TEXT NOT NULL,
    source_group TEXT,
    reference_frame_id TEXT REFERENCES frames(frame_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_split ON frames(split);
CREATE INDEX IF NOT EXISTS idx_frames_venue ON frames(venue_type);
CREATE INDEX IF NOT EXISTS idx_frames_normal ON frames(is_normal);
CREATE INDEX IF NOT EXISTS idx_frames_source_grp ON frames(source_group);

CREATE TABLE IF NOT EXISTS annotations_bbox (
    ann_id INTEGER PRIMARY KEY,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    label_class TEXT NOT NULL,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL,
    txt_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_bbox_frame ON annotations_bbox(frame_id);

CREATE TABLE IF NOT EXISTS annotations_masks (
    mask_id INTEGER PRIMARY KEY,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    ref_frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    mask_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mask_frame ON annotations_masks(frame_id);

CREATE TABLE IF NOT EXISTS experiments (
    exp_id INTEGER PRIMARY KEY,
    pipeline TEXT NOT NULL CHECK (pipeline IN ('P1','P2','P3','P4')),
    model_variant TEXT NOT NULL,
    dataset_filter JSON,
    hyperparams JSON,
    run_date DATETIME NOT NULL DEFAULT (datetime('now')),
    artifact_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('running','done','failed'))
        DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS results (
    result_id INTEGER PRIMARY KEY,
    exp_id INTEGER NOT NULL REFERENCES experiments(exp_id),
    fold TEXT,
    venue_type TEXT CHECK (venue_type IN ('porta','corridoio','scala')),
    mAP50 REAL,
    mAP50_95 REAL,
    precision REAL,
    recall REAL,
    F1 REAL,
    auroc REAL,
    pro_score REAL,
    anomaly_threshold REAL,
    latency_ms REAL,
    false_alarm_rate REAL,
    false_negative_rate REAL
);

CREATE INDEX IF NOT EXISTS idx_results_exp ON results(exp_id);
