CREATE TABLE IF NOT EXISTS env_configs (
	id BIGINT NOT NULL AUTO_INCREMENT,
	name VARCHAR(128) NOT NULL,
	preset_name VARCHAR(32),
	description TEXT NOT NULL,
	config_json JSON NOT NULL,
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
	updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
	PRIMARY KEY (id)
)ENGINE=InnoDB CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS simulation_snapshots (
	id BIGINT NOT NULL AUTO_INCREMENT,
	name VARCHAR(128) NOT NULL,
	env_config_id BIGINT,
	policy_label VARCHAR(32) NOT NULL,
	seed BIGINT NOT NULL,
	metrics_json JSON NOT NULL,
	trace_json JSON NOT NULL,
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
	PRIMARY KEY (id),
	FOREIGN KEY(env_config_id) REFERENCES env_configs (id) ON DELETE SET NULL
)ENGINE=InnoDB CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS training_runs (
	id VARCHAR(32) NOT NULL,
	run_name VARCHAR(128) NOT NULL,
	model_type VARCHAR(16) NOT NULL,
	env_config_id BIGINT,
	config_json JSON NOT NULL,
	episodes INTEGER NOT NULL,
	current_episode INTEGER NOT NULL,
	seed BIGINT NOT NULL,
	device VARCHAR(16) NOT NULL,
	threads INTEGER NOT NULL,
	status VARCHAR(16) NOT NULL,
	best_score DOUBLE,
	results_dir VARCHAR(512) NOT NULL,
	best_checkpoint TEXT,
	last_checkpoint TEXT,
	error_message TEXT,
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
	finished_at DATETIME(6),
	PRIMARY KEY (id),
	FOREIGN KEY(env_config_id) REFERENCES env_configs (id) ON DELETE SET NULL,
	UNIQUE (results_dir)
)ENGINE=InnoDB CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS evaluation_records (
	id BIGINT NOT NULL AUTO_INCREMENT,
	name VARCHAR(128) NOT NULL,
	method VARCHAR(32) NOT NULL,
	run_id VARCHAR(32),
	checkpoint_path TEXT,
	env_config_id BIGINT,
	seeds_json JSON NOT NULL,
	summary_json JSON NOT NULL,
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
	PRIMARY KEY (id),
	FOREIGN KEY(run_id) REFERENCES training_runs (id) ON DELETE SET NULL,
	FOREIGN KEY(env_config_id) REFERENCES env_configs (id) ON DELETE SET NULL
)ENGINE=InnoDB CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS training_metrics (
	id BIGINT NOT NULL AUTO_INCREMENT,
	run_id VARCHAR(32) NOT NULL,
	episode INTEGER NOT NULL,
	episode_reward DOUBLE,
	objective_score DOUBLE,
	task_success_rate DOUBLE,
	acceptance_rate DOUBLE,
	deadline_miss_rate DOUBLE,
	avg_delay_s DOUBLE,
	avg_energy_j DOUBLE,
	mean_load_variance DOUBLE,
	load_balance_degree DOUBLE,
	mean_hops_accepted DOUBLE,
	mean_path_length DOUBLE,
	loss DOUBLE,
	approx_kl DOUBLE,
	entropy DOUBLE,
	explained_variance DOUBLE,
	created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
	PRIMARY KEY (id),
	UNIQUE (run_id, episode),
	FOREIGN KEY(run_id) REFERENCES training_runs (id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4;
