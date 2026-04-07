from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
    StepResult,
    TaskInfo,
    TaskGradeResult,
)


# ---------------------------------------------------------------------------
# Internal incident definition (not exposed to agent)
# ---------------------------------------------------------------------------

@dataclass
class IncidentData:
    incident_id: str
    title: str
    true_severity: str                        # P0 | P1 | P2 | P3
    true_root_cause_service: str              # exact service name
    root_cause_family: str                    # database | cache | config | service | network | resource
    affected_users: int
    error_rate: float
    latency_p99_ms: int
    logs: List[str]                           # intentionally includes misleading lines
    metrics: Dict[str, float]
    available_runbooks: List[Dict[str, str]]  # 4 options, only one correct
    correct_runbook_id: str
    correct_eta_minutes: int
    customer_impact_keywords: List[str]       # terms for status-update scoring
    severity_language: Dict[str, List[str]]   # good/bad words per severity tier
    # Per-incident specific evidence terms that can ONLY come from reading logs/metrics
    # A model that just echoes the service name back does NOT get these
    evidence_signals: List[str] = field(default_factory=list)  # populated per incident


# ---------------------------------------------------------------------------
# Incident corpus — 12 real-world incidents with misleading signals
# ---------------------------------------------------------------------------

def _incidents() -> List[IncidentData]:
    return [
        # ── INC-001: DB connection pool exhausted ──────────────────────────
        # Misleading: all surface errors are in payment-svc, root cause is postgres
        IncidentData(
            incident_id="INC-001",
            title="Checkout failures — payment service elevated 5xx rate",
            true_severity="P1",
            true_root_cause_service="postgres-payments",
            root_cause_family="database",
            affected_users=8400,
            error_rate=0.34,
            latency_p99_ms=4200,
            logs=[
                "[payment-svc] ERROR: Connection timeout after 3000ms for order #88421",
                "[payment-svc] ERROR: Failed to process charge — transaction aborted",
                "[payment-svc] WARN:  Retry 3/3 exhausted for tx_id=tx_92k11",
                "[payment-svc] ERROR: java.sql.SQLTimeoutException: query exceeded 3s limit",
                "[postgres-payments] WARN:  connection pool exhausted (active=50 waiting=127)",
                "[postgres-payments] ERROR: remaining connection slots reserved for replication",
                "[load-balancer] INFO:  payment-svc health checks passing (pods healthy)",
                "[alertmanager] FIRED: PaymentErrorRateHigh threshold=0.05 current=0.34",
            ],
            metrics={
                "payment_svc.error_rate": 0.34,
                "payment_svc.latency_p99_ms": 4200,
                "payment_svc.pod_restarts_1h": 0,
                "postgres_payments.connections_active": 50,
                "postgres_payments.connections_waiting": 127,
                "postgres_payments.cpu_percent": 11.0,
                "postgres_payments.disk_io_util": 0.18,
            },
            available_runbooks=[
                {"id": "RB-007", "description": "Restart payment-service pods and clear in-flight transaction queue"},
                {"id": "RB-019", "description": "Scale postgres connection pool limit and terminate long-running idle connections"},
                {"id": "RB-033", "description": "Enable payment circuit breaker and redirect traffic to standby region"},
                {"id": "RB-041", "description": "Roll back payment-service deployment to previous stable release"},
            ],
            correct_runbook_id="RB-019",
            correct_eta_minutes=18,
            customer_impact_keywords=["checkout", "payment", "order", "purchase"],
            severity_language={
                "good": ["investigating", "aware", "working to resolve", "affected users"],
                "bad": ["minor", "all users", "complete outage", "critical emergency"],
            },
            evidence_signals=[
                "connection pool", "pool exhausted", "waiting=127", "active=50",
                "replication slot", "connection slots reserved", "sqltimeoutexception",
            ],
        ),

        # ── INC-002: Redis session store OOM ───────────────────────────────
        # Misleading: auth-service shows 401/500s; root cause is Redis OOM
        IncidentData(
            incident_id="INC-002",
            title="Login failures across all products — authentication service degraded",
            true_severity="P1",
            true_root_cause_service="redis-sessions",
            root_cause_family="cache",
            affected_users=21000,
            error_rate=0.61,
            latency_p99_ms=8900,
            logs=[
                "[auth-svc] ERROR: Session validation failed for user_id=u_4482k",
                "[auth-svc] ERROR: RedisCommandTimeoutException after 500ms",
                "[auth-svc] WARN:  Falling back to DB session lookup (redis unavailable)",
                "[auth-svc] ERROR: DB session lookup overloaded — rejecting request",
                "[redis-sessions] WARN:  maxmemory policy=allkeys-lru eviction_rate=12000/s",
                "[redis-sessions] ERROR: OOM command not allowed when used memory > maxmemory",
                "[redis-sessions] INFO:  used_memory=7.98GB maxmemory=8GB",
                "[auth-svc] INFO:  pod health: running=8/8 cpu=42% — pods are healthy",
            ],
            metrics={
                "auth_svc.error_rate": 0.61,
                "auth_svc.latency_p99_ms": 8900,
                "auth_svc.pod_count_healthy": 8,
                "redis_sessions.used_memory_gb": 7.98,
                "redis_sessions.maxmemory_gb": 8.0,
                "redis_sessions.eviction_rate_per_s": 12000,
                "redis_sessions.connected_clients": 892,
            },
            available_runbooks=[
                {"id": "RB-002", "description": "Scale auth-service pods horizontally to handle elevated request load"},
                {"id": "RB-011", "description": "Flush expired Redis sessions and increase Redis maxmemory allocation"},
                {"id": "RB-024", "description": "Enable auth-service maintenance mode and redirect to SSO fallback"},
                {"id": "RB-038", "description": "Restart Redis cluster with persistence disabled to clear memory"},
            ],
            correct_runbook_id="RB-011",
            correct_eta_minutes=10,
            customer_impact_keywords=["login", "sign in", "account access", "authentication"],
            severity_language={
                "good": ["investigating", "login issues", "aware", "working to resolve"],
                "bad": ["minor", "all systems down", "complete outage"],
            },
            evidence_signals=[
                "maxmemory", "oom command", "used_memory=7.98", "eviction_rate=12000",
                "allkeys-lru", "memory > maxmemory", "rediscommandtimeout",
            ],
        ),

        # ── INC-003: DNS misconfiguration — P0 complete outage ─────────────
        # Misleading: all services show errors; root cause is DNS change during maintenance
        IncidentData(
            incident_id="INC-003",
            title="COMPLETE OUTAGE — all services unreachable after network maintenance",
            true_severity="P0",
            true_root_cause_service="dns-resolver",
            root_cause_family="network",
            affected_users=180000,
            error_rate=0.99,
            latency_p99_ms=30000,
            logs=[
                "[api-gateway] ERROR: upstream connect error — connection refused for payment-svc",
                "[api-gateway] ERROR: upstream connect error — connection refused for auth-svc",
                "[api-gateway] ERROR: upstream connect error — connection refused for catalog-svc",
                "[k8s-node-01] WARN:  DNS resolution failure for payment-svc.prod.svc.cluster.local",
                "[k8s-node-02] WARN:  DNS resolution failure for auth-svc.prod.svc.cluster.local",
                "[coredns] ERROR: forward loop detected — 10.96.0.10 → 10.96.0.10",
                "[network-team] NOTE: routine DNS config update applied at 14:32 UTC",
                "[alertmanager] FIRED: AllServicesDown severity=critical",
            ],
            metrics={
                "api_gateway.error_rate": 0.99,
                "api_gateway.requests_per_s": 4200,
                "dns_resolution.failure_rate": 0.98,
                "coredns.loop_detected": 1.0,
                "k8s.pods_ready_percent": 0.97,  # pods are healthy — misleading
                "network.packet_loss_percent": 0.01,
            },
            available_runbooks=[
                {"id": "RB-003", "description": "Rolling restart all application pods to re-establish upstream connections"},
                {"id": "RB-016", "description": "Revert DNS configuration to last known-good snapshot and restart CoreDNS"},
                {"id": "RB-029", "description": "Scale API gateway and enable request buffering to absorb upstream failures"},
                {"id": "RB-044", "description": "Activate disaster recovery site and update global load balancer routing"},
            ],
            correct_runbook_id="RB-016",
            correct_eta_minutes=25,
            customer_impact_keywords=["all services", "unavailable", "outage", "unable to access"],
            severity_language={
                "good": ["all users", "complete outage", "working urgently", "critical", "immediately"],
                "bad": ["some users", "intermittent", "minor", "investigating"],
            },
            evidence_signals=[
                "forward loop", "loop detected", "coredns", "dns resolution failure",
                "10.96.0.10", "dns config update", "14:32 utc",
            ],
        ),

        # ── INC-004: TLS cert expiry on internal webhook endpoint ──────────
        IncidentData(
            incident_id="INC-004",
            title="Webhook delivery failures — third-party integrations broken",
            true_severity="P2",
            true_root_cause_service="tls-cert-manager",
            root_cause_family="config",
            affected_users=340,
            error_rate=1.0,
            latency_p99_ms=200,
            logs=[
                "[webhook-svc] ERROR: TLS handshake failed for endpoint https://hooks.partner-a.com",
                "[webhook-svc] ERROR: certificate verify failed — certificate has expired",
                "[webhook-svc] INFO:  Retry 3/3 for event_id=evt_8821 — giving up",
                "[webhook-svc] ERROR: certificate_expiry_check: cert expired 6 hours ago",
                "[tls-cert-manager] WARN:  Certificate renewal missed for webhook-internal.prod",
                "[tls-cert-manager] ERROR: ACME challenge failed — cert-manager pod restarted before challenge",
                "[webhook-svc] INFO:  webhook-svc pods healthy, cpu=4%, mem=210MB",
                "[alertmanager] FIRED: WebhookDeliveryRate0 duration=6h",
            ],
            metrics={
                "webhook_svc.delivery_success_rate": 0.0,
                "webhook_svc.delivery_failures_1h": 1842,
                "webhook_svc.pod_restarts": 0,
                "tls_cert_manager.cert_expiry_days": -0.25,  # expired 6h ago
                "tls_cert_manager.renewal_failures": 1,
            },
            available_runbooks=[
                {"id": "RB-005", "description": "Restart webhook-service and clear the delivery retry queue"},
                {"id": "RB-014", "description": "Manually renew TLS certificate using certbot and redeploy cert-manager"},
                {"id": "RB-027", "description": "Disable TLS verification on webhook endpoints (temporary workaround)"},
                {"id": "RB-039", "description": "Roll back cert-manager to previous version and re-trigger certificate issuance"},
            ],
            correct_runbook_id="RB-014",
            correct_eta_minutes=20,
            customer_impact_keywords=["webhook", "integration", "third-party", "notification"],
            severity_language={
                "good": ["investigating", "aware", "webhook", "integration", "working to restore"],
                "bad": ["all users", "complete outage", "critical emergency"],
            },
            evidence_signals=[
                "cert expired", "certificate has expired", "acme challenge failed",
                "renewal missed", "expiry_days=-0.25", "cert-manager pod restarted",
            ],
        ),

        # ── INC-005: Kafka partition leader rebalancing ────────────────────
        IncidentData(
            incident_id="INC-005",
            title="Data pipeline delays — event processing backlog growing",
            true_severity="P2",
            true_root_cause_service="kafka",
            root_cause_family="service",
            affected_users=0,
            error_rate=0.0,
            latency_p99_ms=0,
            logs=[
                "[pipeline-worker] WARN:  Consumer lag on topic=order-events partition=3 lag=824000",
                "[pipeline-worker] WARN:  Consumer lag on topic=payment-events partition=1 lag=412000",
                "[kafka-broker-2] WARN:  Partition leader election in progress for 14 partitions",
                "[kafka-broker-2] INFO:  Broker restarted after node replacement at 09:14 UTC",
                "[kafka-broker-1] WARN:  Under-replicated partitions: 14 (expected 0)",
                "[pipeline-worker] INFO:  Workers healthy — waiting for messages to be available",
                "[alertmanager] FIRED: KafkaConsumerLagHigh topic=order-events lag=824000",
            ],
            metrics={
                "kafka.under_replicated_partitions": 14,
                "kafka.partition_leader_elections": 14,
                "kafka.broker_count": 3,
                "pipeline.consumer_lag_order_events": 824000,
                "pipeline.consumer_lag_payment_events": 412000,
                "pipeline.worker_pod_count_healthy": 12,
            },
            available_runbooks=[
                {"id": "RB-006", "description": "Scale pipeline worker pods to increase consumer throughput"},
                {"id": "RB-017", "description": "Trigger preferred replica leader election to rebalance Kafka partitions"},
                {"id": "RB-031", "description": "Restart Kafka broker-2 and force full ISR resync"},
                {"id": "RB-043", "description": "Pause non-critical consumers and prioritize order-events topic processing"},
            ],
            correct_runbook_id="RB-017",
            correct_eta_minutes=12,
            customer_impact_keywords=["data processing", "pipeline", "delays", "event"],
            severity_language={
                "good": ["investigating", "processing delays", "no direct user impact", "monitoring"],
                "bad": ["users affected", "outage", "critical", "immediate action"],
            },
            evidence_signals=[
                "partition leader election", "under-replicated partitions", "broker restarted",
                "consumer lag", "lag=824000", "node replacement", "09:14 utc",
            ],
        ),

        # ── INC-006: Elasticsearch index corruption ────────────────────────
        IncidentData(
            incident_id="INC-006",
            title="Search returning empty results — product catalog search broken",
            true_severity="P1",
            true_root_cause_service="elasticsearch",
            root_cause_family="service",
            affected_users=15200,
            error_rate=0.0,
            latency_p99_ms=120,
            logs=[
                "[search-svc] WARN:  Query returned 0 results for term='shoes' (expected >500)",
                "[search-svc] INFO:  Elasticsearch connection healthy — cluster status green",
                "[elasticsearch] ERROR: shard [catalog_v2][3] ShardRecoveryException — index corrupt",
                "[elasticsearch] WARN:  Index catalog_v2 missing 3 of 5 primary shards after migration",
                "[elasticsearch] INFO:  Migration job completed at 03:22 UTC — exit code 0",
                "[search-svc] INFO:  search-svc pods all healthy, no restarts",
                "[alertmanager] FIRED: SearchResultsEmpty duration=45m",
            ],
            metrics={
                "search_svc.results_per_query_avg": 0.0,
                "search_svc.error_rate": 0.01,
                "elasticsearch.cluster_status": 1.0,  # 1=green (misleading)
                "elasticsearch.shards_active": 12,
                "elasticsearch.shards_unassigned": 3,
                "elasticsearch.index_catalog_v2_docs": 0,
            },
            available_runbooks=[
                {"id": "RB-008", "description": "Restart search-service pods and clear the query cache"},
                {"id": "RB-021", "description": "Restore Elasticsearch index from latest snapshot and re-run migration validation"},
                {"id": "RB-034", "description": "Switch search traffic to secondary read replica cluster"},
                {"id": "RB-042", "description": "Force re-index from primary database source into a fresh index alias"},
            ],
            correct_runbook_id="RB-021",
            correct_eta_minutes=40,
            customer_impact_keywords=["search", "product", "catalog", "results"],
            severity_language={
                "good": ["investigating", "search", "working to restore", "aware"],
                "bad": ["all services down", "complete outage", "minor"],
            },
            evidence_signals=[
                "shardrecoveryexception", "index corrupt", "missing 3 of 5 primary shards",
                "migration job", "03:22 utc", "shards_unassigned", "catalog_v2",
            ],
        ),

        # ── INC-007: SMTP relay rate limit ────────────────────────────────
        # Misleading: notification-svc shows 500s but it's the SMTP relay being throttled
        IncidentData(
            incident_id="INC-007",
            title="Email notifications not delivering — transactional emails failing",
            true_severity="P2",
            true_root_cause_service="smtp-relay",
            root_cause_family="service",
            affected_users=5600,
            error_rate=0.89,
            latency_p99_ms=12000,
            logs=[
                "[notification-svc] ERROR: Failed to send email for event=order_confirmed user=u_8821",
                "[notification-svc] ERROR: SMTPException: 421 Too many connections from your IP",
                "[notification-svc] WARN:  Email queue depth: 18400 (normal: <200)",
                "[smtp-relay] ERROR: rate limit exceeded — 1000 emails/min limit hit",
                "[smtp-relay] WARN:  Throttling sender: notifications@company.com",
                "[smtp-relay] INFO:  Hourly quota: 42000/50000 used (84%)",
                "[notification-svc] INFO:  notification-svc pods healthy — processing queue",
                "[alertmanager] FIRED: EmailDeliveryRateHigh threshold=0.10 current=0.89",
            ],
            metrics={
                "notification_svc.email_error_rate": 0.89,
                "notification_svc.queue_depth": 18400,
                "notification_svc.pod_count_healthy": 6,
                "smtp_relay.rate_limit_hits_1h": 144,
                "smtp_relay.hourly_quota_used": 42000,
                "smtp_relay.hourly_quota_max": 50000,
            },
            available_runbooks=[
                {"id": "RB-009", "description": "Scale notification-service horizontally to drain the email queue faster"},
                {"id": "RB-022", "description": "Switch SMTP relay to backup provider and throttle send rate to avoid re-triggering limits"},
                {"id": "RB-035", "description": "Pause low-priority marketing emails and prioritize transactional notifications only"},
                {"id": "RB-046", "description": "Restart SMTP relay service and clear the connection pool"},
            ],
            correct_runbook_id="RB-022",
            correct_eta_minutes=30,
            customer_impact_keywords=["email", "notification", "transactional", "confirmation"],
            severity_language={
                "good": ["investigating", "email delays", "aware", "working to restore"],
                "bad": ["all users", "complete outage", "critical"],
            },
            evidence_signals=[
                "rate limit exceeded", "1000 emails/min", "throttling sender",
                "421 too many connections", "hourly quota", "42000/50000", "queue depth",
            ],
        ),

        # ── INC-008: Memcached eviction spike causing DB overload ──────────
        IncidentData(
            incident_id="INC-008",
            title="User profile pages loading slowly — API latency spike",
            true_severity="P2",
            true_root_cause_service="memcached",
            root_cause_family="cache",
            affected_users=31000,
            error_rate=0.04,
            latency_p99_ms=6800,
            logs=[
                "[profile-api] WARN:  p99 latency 6800ms (SLO=500ms)",
                "[profile-api] WARN:  Cache miss rate: 94% (normal: 8%)",
                "[profile-api] WARN:  Falling back to DB for user_id=u_* (high volume)",
                "[postgres-users] WARN:  CPU at 89% — query queue depth 420",
                "[memcached] WARN:  Eviction rate: 48000 items/s (normal: 200/s)",
                "[memcached] INFO:  Memory usage: 15.9GB / 16GB — at capacity",
                "[profile-api] INFO:  All pods healthy (16/16 running)",
                "[alertmanager] FIRED: ProfileAPILatencyHigh p99=6800ms",
            ],
            metrics={
                "profile_api.latency_p99_ms": 6800,
                "profile_api.error_rate": 0.04,
                "profile_api.cache_miss_rate": 0.94,
                "memcached.eviction_rate_per_s": 48000,
                "memcached.memory_used_gb": 15.9,
                "memcached.memory_max_gb": 16.0,
                "postgres_users.cpu_percent": 89.0,
            },
            available_runbooks=[
                {"id": "RB-010", "description": "Scale profile-api pods to reduce per-pod load"},
                {"id": "RB-023", "description": "Expand memcached cluster capacity and adjust eviction policy to allkeys-lfu"},
                {"id": "RB-036", "description": "Add read replica to postgres-users and distribute query load"},
                {"id": "RB-048", "description": "Enable response caching at API gateway layer for profile endpoints"},
            ],
            correct_runbook_id="RB-023",
            correct_eta_minutes=15,
            customer_impact_keywords=["profile", "account", "slow loading", "performance"],
            severity_language={
                "good": ["investigating", "performance", "slow", "aware", "working to resolve"],
                "bad": ["all users", "outage", "critical emergency"],
            },
            evidence_signals=[
                "eviction rate", "48000 items/s", "cache miss rate: 94%", "at capacity",
                "15.9gb / 16gb", "allkeys-lru", "miss_rate=0.94",
            ],
        ),

        # ── INC-009: ML inference service cascade ─────────────────────────
        # Misleading: API gateway shows 503s; root cause is ML service GPU OOM
        IncidentData(
            incident_id="INC-009",
            title="Recommendation service degraded — homepage personalization failing",
            true_severity="P2",
            true_root_cause_service="ml-inference",
            root_cause_family="resource",
            affected_users=62000,
            error_rate=0.78,
            latency_p99_ms=15000,
            logs=[
                "[api-gateway] ERROR: upstream timeout for /api/recommendations (15s)",
                "[api-gateway] WARN:  Circuit breaker OPEN for ml-inference-svc",
                "[ml-inference] ERROR: CUDA out of memory — batch_size=64 model=rec-transformer-v3",
                "[ml-inference] ERROR: RuntimeError: CUDA error: device-side assert triggered",
                "[ml-inference] WARN:  GPU memory: 39.8GB / 40GB used",
                "[ml-inference] INFO:  Attempting to serve with reduced batch_size=8 — failing",
                "[api-gateway] INFO:  Serving fallback static recommendations for affected users",
                "[alertmanager] FIRED: RecommendationServiceDown duration=22m",
            ],
            metrics={
                "api_gateway.recommendation_error_rate": 0.78,
                "ml_inference.gpu_memory_used_gb": 39.8,
                "ml_inference.gpu_memory_max_gb": 40.0,
                "ml_inference.request_queue_depth": 8200,
                "ml_inference.pod_restarts_1h": 14,
                "api_gateway.pod_count_healthy": 12,
            },
            available_runbooks=[
                {"id": "RB-004", "description": "Scale API gateway and increase upstream timeout to 30s"},
                {"id": "RB-015", "description": "Reduce ML inference batch size and restart pods with GPU memory limit enforcement"},
                {"id": "RB-028", "description": "Roll back recommendation model to previous version with lower memory footprint"},
                {"id": "RB-045", "description": "Enable static recommendation fallback permanently and schedule ML fix for off-peak"},
            ],
            correct_runbook_id="RB-028",
            correct_eta_minutes=35,
            customer_impact_keywords=["recommendation", "homepage", "personalization", "suggestions"],
            severity_language={
                "good": ["investigating", "personalization", "aware", "working to resolve"],
                "bad": ["all services", "complete outage", "critical emergency"],
            },
            evidence_signals=[
                "cuda out of memory", "gpu memory: 39.8gb", "batch_size=64", "rec-transformer-v3",
                "pod_restarts_1h=14", "device-side assert", "gpu_memory_used",
            ],
        ),

        # ── INC-010: CDN cache invalidation crash after deploy ────────────
        IncidentData(
            incident_id="INC-010",
            title="Stale content served — users seeing outdated product prices and images",
            true_severity="P1",
            true_root_cause_service="cache-invalidation",
            root_cause_family="service",
            affected_users=98000,
            error_rate=0.0,
            latency_p99_ms=80,
            logs=[
                "[cdn] INFO:  Cache hit rate: 99.2% (normal, high performance)",
                "[catalog-api] INFO:  Price update published for 14000 SKUs at 11:02 UTC",
                "[cache-invalidation-svc] ERROR: NullPointerException in CDNPurgeJob.run()",
                "[cache-invalidation-svc] ERROR: Job failed — 0 CDN purge requests sent",
                "[cache-invalidation-svc] INFO:  Deploy v2.4.1 rolled out at 11:00 UTC",
                "[cdn] INFO:  No purge requests received since 10:58 UTC",
                "[catalog-api] INFO:  API healthy — serving correct prices from DB",
                "[alertmanager] FIRED: CDNStalePriceDetected stale_skus=14000",
            ],
            metrics={
                "cdn.cache_hit_rate": 0.992,
                "cdn.purge_requests_last_1h": 0,
                "cache_invalidation_svc.job_failures_1h": 48,
                "cache_invalidation_svc.job_success_rate": 0.0,
                "catalog_api.error_rate": 0.002,
                "catalog_api.price_update_events_queued": 14000,
            },
            available_runbooks=[
                {"id": "RB-001", "description": "Manually trigger CDN purge for all product catalog paths and images"},
                {"id": "RB-012", "description": "Roll back cache-invalidation-svc to v2.4.0 and trigger full cache purge"},
                {"id": "RB-026", "description": "Set CDN cache TTL to 0 for catalog paths and redeploy CDN configuration"},
                {"id": "RB-040", "description": "Restart catalog-api to force re-fetch and bypass CDN cache"},
            ],
            correct_runbook_id="RB-012",
            correct_eta_minutes=20,
            customer_impact_keywords=["product", "price", "stale", "content", "outdated"],
            severity_language={
                "good": ["investigating", "aware", "incorrect prices", "working to resolve"],
                "bad": ["all services", "outage", "critical emergency"],
            },
            evidence_signals=[
                "nullpointerexception", "cdnpurgejob.run", "0 cdn purge requests",
                "deploy v2.4.1", "11:00 utc", "no purge requests since 10:58",
                "job_success_rate=0.0",
            ],
        ),

        # ── INC-011: P3 Gradual memory leak (non-urgent) ──────────────────
        IncidentData(
            incident_id="INC-011",
            title="Recommendation pods slowly increasing memory — approaching limit",
            true_severity="P3",
            true_root_cause_service="recommendation-svc",
            root_cause_family="resource",
            affected_users=0,
            error_rate=0.002,
            latency_p99_ms=310,
            logs=[
                "[recommendation-svc] INFO:  Memory: 3.1GB / 4GB (pod-1), 3.0GB / 4GB (pod-2)",
                "[recommendation-svc] WARN:  Memory growth trend: +180MB/hour (last 12h)",
                "[recommendation-svc] INFO:  No errors — service operating normally",
                "[recommendation-svc] INFO:  Feature flag 'enable_user_history_cache' enabled 13h ago",
                "[k8s] WARN:  recommendation-svc-pod-1 approaching memory limit (78%)",
                "[alertmanager] FIRED: PodMemoryHigh threshold=75% current=78%",
            ],
            metrics={
                "recommendation_svc.memory_used_gb_pod1": 3.1,
                "recommendation_svc.memory_limit_gb": 4.0,
                "recommendation_svc.memory_growth_mb_per_hour": 180,
                "recommendation_svc.error_rate": 0.002,
                "recommendation_svc.latency_p99_ms": 310,
            },
            available_runbooks=[
                {"id": "RB-018", "description": "Schedule rolling restart of recommendation-svc pods during next maintenance window"},
                {"id": "RB-030", "description": "Immediately disable 'enable_user_history_cache' feature flag and monitor memory trend"},
                {"id": "RB-037", "description": "Increase pod memory limit to 8GB as a temporary measure"},
                {"id": "RB-049", "description": "Page on-call engineer to investigate and hotfix the memory leak"},
            ],
            correct_runbook_id="RB-030",
            correct_eta_minutes=5,
            customer_impact_keywords=["no direct user impact", "proactive", "monitoring", "potential"],
            severity_language={
                "good": ["proactive", "no current user impact", "monitoring", "aware", "preventive"],
                "bad": ["critical", "all users affected", "immediate outage", "emergency"],
            },
            evidence_signals=[
                "+180mb/hour", "memory growth trend", "enable_user_history_cache",
                "13h ago", "3.1gb / 4gb", "approaching memory limit", "78%",
            ],
        ),

        # ── INC-012: P0 — Billing engine down, revenue stopped ────────────
        IncidentData(
            incident_id="INC-012",
            title="CRITICAL — billing engine stopped processing — revenue collection halted",
            true_severity="P0",
            true_root_cause_service="billing-engine",
            root_cause_family="service",
            affected_users=0,
            error_rate=1.0,
            latency_p99_ms=0,
            logs=[
                "[billing-engine] ERROR: Failed to connect to stripe-gateway: connection refused",
                "[billing-engine] ERROR: Job billing_daily_run failed after 3 attempts",
                "[billing-engine] ERROR: Config: STRIPE_API_ENDPOINT=https://api.stripe-internal.company.com",
                "[billing-engine] WARN:  STRIPE_API_ENDPOINT was changed by config deploy at 02:00 UTC",
                "[stripe-gateway] INFO:  Service healthy — 0 requests received in last 2h",
                "[billing-engine] INFO:  billing-engine pods all running (4/4)",
                "[alertmanager] FIRED: BillingJobFailure severity=critical revenue_at_risk=true",
            ],
            metrics={
                "billing_engine.job_success_rate": 0.0,
                "billing_engine.jobs_failed_2h": 6,
                "billing_engine.pod_count_healthy": 4,
                "stripe_gateway.requests_received_2h": 0,
                "stripe_gateway.error_rate": 0.0,
            },
            available_runbooks=[
                {"id": "RB-013", "description": "Restart billing-engine pods and clear the failed job queue"},
                {"id": "RB-025", "description": "Revert STRIPE_API_ENDPOINT config to production value and trigger job rerun"},
                {"id": "RB-032", "description": "Failover billing processing to manual batch mode and notify finance team"},
                {"id": "RB-047", "description": "Roll back the config deploy from 02:00 UTC to restore billing engine connectivity"},
            ],
            correct_runbook_id="RB-025",
            correct_eta_minutes=8,
            customer_impact_keywords=["billing", "payment processing", "revenue", "charges"],
            severity_language={
                "good": ["critical", "immediate", "all hands", "revenue", "urgently"],
                "bad": ["minor", "some users", "investigating slowly", "monitoring"],
            },
            evidence_signals=[
                "stripe_api_endpoint", "config deploy at 02:00", "stripe-internal.company.com",
                "0 requests received in last 2h", "jobs_failed_2h=6", "billing_daily_run",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _severity_distance(a: str, b: str) -> int:
    return abs(_SEVERITY_ORDER.get(a, 99) - _SEVERITY_ORDER.get(b, 99))


# ---------------------------------------------------------------------------
# OpsEnv — main environment class
# ---------------------------------------------------------------------------

class OpsEnv:
    """
    Production Incident Response environment.

    Three sequential tasks per episode:
    1. classify_severity (easy)   — P0/P1/P2/P3
    2. identify_root_cause (medium) — which service caused the incident
    3. execute_response (hard)    — runbook + ETA + customer status update

    Episode reward max = 3.0 (1.0 per task).
    """

    TASKS: List[str] = ["classify_severity", "identify_root_cause", "execute_response"]

    TASK_INFO: List[TaskInfo] = [
        TaskInfo(
            name="classify_severity",
            difficulty="easy",
            description=(
                "Classify the incident severity as P0, P1, P2, or P3 based on "
                "error rate, affected users, business criticality, and service health signals."
            ),
        ),
        TaskInfo(
            name="identify_root_cause",
            difficulty="medium",
            description=(
                "Identify the root cause service (not the symptom service) by analysing "
                "log patterns and metrics. Provide the service name and a brief reason "
                "citing evidence from the logs."
            ),
        ),
        TaskInfo(
            name="execute_response",
            difficulty="hard",
            description=(
                "Select the correct runbook from the available options, estimate the "
                "time to resolution in minutes, and write a customer-facing status page "
                "update (≥20 words) that mentions the affected functionality without "
                "leaking internal service names."
            ),
        ),
    ]

    def __init__(self, incidents: Optional[List[IncidentData]] = None):
        self._all_incidents = incidents or _incidents()
        self._episode_counter = 0
        self._session_id: Optional[str] = None
        self._task_scores: Dict[str, float] = {}
        self._task_details: Dict[str, Dict[str, Any]] = {}
        # Safe defaults
        self._incident = self._all_incidents[0]
        self.task_index = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        self.completed_tasks: List[str] = []
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        self._episode_counter += 1
        self._session_id = uuid.uuid4().hex[:12]
        idx = (self._episode_counter - 1) % len(self._all_incidents)
        self._incident = self._all_incidents[idx]
        self.task_index = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        self.completed_tasks = []
        self.history = []
        self._task_scores = {}
        self._task_details = {}
        return self._build_result(reward=0.0, info={"event": "reset"})

    def state(self) -> IncidentState:
        return IncidentState(
            episode_id=self._session_id or "",
            step_count=self.step_count,
            total_reward=round(self.total_reward, 4),
            current_task_index=self.task_index,
            current_task_name=self.current_task_name,
            incident_id=self._incident.incident_id,
            completed_tasks=list(self.completed_tasks),
            done=self.done,
            notes={
                "incident_severity_actual": self._incident.true_severity,
                "task_scores": dict(self._task_scores),
                "remaining_tasks": max(0, len(self.TASKS) - self.task_index),
            },
        )

    @property
    def current_task_name(self) -> str:
        if self.done:
            return "completed"
        return self.TASKS[self.task_index]

    def step(self, action: IncidentAction) -> StepResult:
        if self.done:
            return self._build_result(reward=0.0, info={"warning": "episode_already_finished"})

        task_name = self.current_task_name
        reward, details = self._score_action(task_name, action)
        self._task_scores[task_name] = round(reward, 4)
        self._task_details[task_name] = details
        self.total_reward += reward
        self.step_count += 1
        self.completed_tasks.append(task_name)
        self.history.append({
            "task": task_name,
            "action_summary": _action_summary(task_name, action),
            "reward": round(reward, 4),
            "details": details,
        })
        self.task_index += 1
        if self.task_index >= len(self.TASKS):
            self.done = True

        return self._build_result(
            reward=reward,
            info={
                "task": task_name,
                "details": details,
                "incident_id": self._incident.incident_id,
                "task_score": round(reward, 4),
            },
        )

    def grade_task(self, task_name: str) -> TaskGradeResult:
        if task_name not in self._task_scores:
            return TaskGradeResult(
                task_name=task_name,
                score=0.0,
                graded=False,
                details={"error": "task_not_completed_yet"},
            )
        return TaskGradeResult(
            task_name=task_name,
            score=self._task_scores[task_name],
            graded=True,
            details=self._task_details.get(task_name, {}),
        )

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------

    def _score_action(self, task_name: str, action: IncidentAction) -> Tuple[float, Dict[str, Any]]:
        if task_name == "classify_severity":
            return self._grade_severity(action.severity)
        if task_name == "identify_root_cause":
            return self._grade_root_cause(action.root_cause_service, action.root_cause_reason)
        if task_name == "execute_response":
            return self._grade_response(action.runbook_id, action.eta_minutes, action.status_update)
        return 0.0, {"error": "unknown_task"}

    def _grade_severity(self, predicted: str) -> Tuple[float, Dict[str, Any]]:
        """
        Exact match  = 1.0
        One level off = 0.35  (reduced from 0.5 — prevents always-P1 gaming)
        Two+ levels off = 0.0

        Severity boundary notes:
          P0↔P1: missing a complete outage is a serious error (0.35)
          P2↔P3: over/under-triaging a minor issue (0.35)
        """
        predicted_norm = (predicted or "").upper().strip()
        true = self._incident.true_severity
        dist = _severity_distance(predicted_norm, true)
        if dist == 0:
            return 1.0, {"match": "exact", "predicted": predicted_norm, "expected": true}
        if dist == 1:
            return 0.35, {"match": "adjacent", "predicted": predicted_norm, "expected": true}
        return 0.0, {"match": "wrong", "predicted": predicted_norm, "expected": true}

    def _grade_root_cause(self, service: str, reason: str) -> Tuple[float, Dict[str, Any]]:
        """
        Exact service + incident-specific evidence cited in reason = 1.00
        Exact service + only service name echoed (no specific signals) = 0.80
        Exact service + reason is empty or <10 chars                    = 0.65
        Same failure family but wrong service                           = 0.40
        Completely wrong                                                = 0.00

        Evidence check requires incident-specific log/metric terms that
        CANNOT be guessed — only obtained by reading the provided logs/metrics.
        Echoing back the service name alone does NOT count as evidence.
        """
        svc_norm = _normalize(service)
        true_svc = self._incident.true_root_cause_service
        true_family = self._incident.root_cause_family
        details: Dict[str, Any] = {
            "predicted_service": svc_norm,
            "expected_service": true_svc,
        }

        if svc_norm == _normalize(true_svc):
            details["service_match"] = "exact"
            reason_norm = _normalize(reason)

            # Require incident-specific evidence signals (not just the service name)
            evidence_signals = self._incident.evidence_signals or []
            # Strip the service name itself from the check — echoing it back doesn't count
            svc_tokens = set(_normalize(true_svc).replace("-", " ").split())
            signal_hits = [
                sig for sig in evidence_signals
                if sig in reason_norm and not all(tok in svc_tokens for tok in sig.split())
            ]

            if signal_hits:
                details["evidence_cited"] = True
                details["evidence_terms_found"] = signal_hits[:3]
                return 1.0, details
            elif len(reason_norm) >= 10:
                # Has a reason but no specific signals — partial credit
                details["evidence_cited"] = False
                details["evidence_note"] = "reason too generic — cite specific log/metric signals"
                return 0.80, details
            else:
                # Reason is empty or trivial
                details["evidence_cited"] = False
                details["evidence_note"] = "no reason provided"
                return 0.65, details

        # Check family match
        predicted_family = _infer_family(svc_norm)
        if predicted_family == true_family:
            details["service_match"] = "family"
            details["predicted_family"] = predicted_family
            details["expected_family"] = true_family
            return 0.40, details

        details["service_match"] = "wrong"
        return 0.0, details

    def _grade_response(
        self, runbook_id: str, eta_minutes: int, status_update: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Runbook correct                                    = +0.50
        ETA within ±25% of expected                       = +0.15  (tightened from ±40%)
        Status update quality (max 0.35):
          Word count ≥ 30                                  = prerequisite (scales if short)
          Mentions customer-facing impact area             = +0.12
          No internal service/infra name leakage           = +0.12
          Appropriate severity tone + action verb          = +0.11
          Over-promising specific ETA in text              = −0.15 penalty  (increased)
          Copy-paste of incident title (lazy)              = −0.05 penalty
          Technical jargon in customer comms               = −0.08 penalty  (new)
        Total max = 1.0

        Hardening rationale (vs original):
        - ETA ±25%: weak models say "5 min" or "60 min"; real ETAs are specific (8–40min)
        - 30-word minimum: forces substantive status updates, not one-liners
        - Leakage list expanded: catches "database", "cache layer", "broker", "cluster",
          "container", "node", "shard", "worker" — common LLM infrastructure words
        - Over-promise penalty −0.15: #1 LLM failure mode, deserves heavier deduction
        - Jargon penalty: penalises "RCA", "SLA", "RTO", "MTTR", "TTR" in customer text
          (internal ops language that should never appear in a public status page)
        """
        score = 0.0
        details: Dict[str, Any] = {}
        inc = self._incident

        # ── Runbook ───────────────────────────────────────────────────
        runbook_correct = runbook_id.strip().upper() == inc.correct_runbook_id.upper()
        if runbook_correct:
            score += 0.50
            details["runbook"] = "correct"
        else:
            details["runbook"] = f"wrong (expected {inc.correct_runbook_id}, got {runbook_id.strip()})"

        # ── ETA (±25% tolerance — tightened from ±40%) ────────────────
        expected_eta = inc.correct_eta_minutes
        if expected_eta > 0 and eta_minutes > 0:
            ratio = eta_minutes / expected_eta
            if 0.75 <= ratio <= 1.25:
                score += 0.15
                details["eta"] = f"ok ({eta_minutes}min vs expected ~{expected_eta}min)"
            else:
                details["eta"] = f"off — {eta_minutes}min vs expected ~{expected_eta}min (need ±25%)"
        else:
            details["eta"] = "not_provided"

        # ── Status update ─────────────────────────────────────────────
        text = status_update.strip().lower()
        word_count = len(text.split())
        details["status_word_count"] = word_count

        # Word count prerequisite — 30 words for full credit (raised from 20)
        # Scales proportionally below threshold so very short updates still earn partial credit
        word_count_multiplier = min(1.0, word_count / 30) if word_count > 0 else 0.0

        update_score = 0.0

        # +0.12: mentions customer-facing impact area
        if any(k in text for k in inc.customer_impact_keywords):
            update_score += 0.12
            details["comms_impact_area"] = True
        else:
            details["comms_impact_area"] = False

        # +0.12: does NOT leak internal service/infra names
        # Expanded list catches more LLM failure modes
        internal_leakage_terms = [
            # Specific services
            "postgres", "redis", "kafka", "elasticsearch", "memcached",
            "coredns", "smtp-relay", "tls-cert", "billing-engine",
            "ml-inference", "cache-invalidation", "cert-manager",
            # Generic infra vocabulary that should never appear in customer comms
            "nginx", "kubernetes", "k8s", "-svc", " pod ", "replica",
            "database", "cache layer", "message broker", "broker",
            "cluster", "container", "docker", " node ", "shard",
            "worker pod", "deployment", "namespace", "ingress",
        ]
        has_leakage = any(t in text for t in internal_leakage_terms)
        if not has_leakage:
            update_score += 0.12
            details["comms_no_leakage"] = True
        else:
            leaked = [t for t in internal_leakage_terms if t in text]
            details["comms_no_leakage"] = False
            details["comms_leaked_terms"] = leaked[:3]

        # +0.11: appropriate tone + contains an action verb
        good_words = inc.severity_language.get("good", [])
        bad_words = inc.severity_language.get("bad", [])
        action_verbs = [
            "investigating", "working", "monitoring", "resolved",
            "aware", "identified", "restoring", "reviewing", "addressing",
            "mitigating", "remediating", "deploying", "rolled back",
        ]
        has_tone = any(w in text for w in good_words)
        has_action_verb = any(v in text for v in action_verbs)
        has_bad_tone = any(w in text for w in bad_words)
        if has_tone and has_action_verb and not has_bad_tone:
            update_score += 0.11
            details["comms_tone"] = "appropriate"
        elif has_bad_tone:
            details["comms_tone"] = "mismatched_severity"
        else:
            details["comms_tone"] = "missing_action_verb_or_tone"

        # −0.15 penalty: over-promising a specific fix time (increased from −0.10)
        # This is the single most common LLM failure mode: "will be fixed in 15 minutes"
        over_promise_pattern = re.search(
            r"(will be (fixed|resolved|back|restored|up) in \d|"
            r"fix(ed)? within \d|"
            r"restor(ed)? (in|within) \d|"
            r"expect.{0,20}(minute|hour)|"
            r"\beta\b.{0,5}\d+\s*(min|hour)|"
            r"back (up |online )?(in|within) \d)",
            text,
        )
        if over_promise_pattern:
            update_score -= 0.15
            details["comms_over_promise_penalty"] = True

        # −0.05 penalty: copy-pasting the incident title verbatim (lazy response)
        title_words = set(_normalize(inc.title).split())
        text_words = set(text.split())
        overlap_ratio = len(title_words & text_words) / max(len(title_words), 1)
        if overlap_ratio > 0.70:
            update_score -= 0.05
            details["comms_title_copy_penalty"] = True

        # −0.08 penalty: internal ops jargon in a customer-facing status page
        # Real public status pages never say "RCA", "RTO", "SLA breach", "MTTR", "TTR"
        ops_jargon_terms = [
            r"\brca\b", r"\brto\b", r"\brpo\b", r"\bmttr\b", r"\bttr\b",
            r"\bsla breach\b", r"\bslo\b", r"\bp99\b", r"\bp95\b",
            r"\blatency spike\b", r"\berror rate\b", r"\bmetrics\b",
            r"\balert(manager|ing)?\b", r"\brollback\b", r"\bhotfix\b",
        ]
        has_jargon = any(re.search(p, text) for p in ops_jargon_terms)
        if has_jargon:
            update_score -= 0.08
            details["comms_jargon_penalty"] = True
            details["comms_jargon_note"] = "ops jargon (RCA/SLA/MTTR/latency/error rate) should not appear in customer comms"

        # Apply word-count multiplier AFTER all sub-scores (including penalties) are summed
        update_score = update_score * word_count_multiplier
        update_score = max(0.0, min(0.35, update_score))
        score += update_score
        details["comms_score"] = round(update_score, 4)

        return round(min(1.0, score), 4), details

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> IncidentObservation:
        inc = self._incident
        if self.done:
            task_name = "completed"
            task_index = len(self.TASKS)
            remaining = 0
            hint = "Episode complete. All 3 tasks finished. Use state() for final scores."
        else:
            task_name = self.current_task_name
            task_index = self.task_index
            remaining = len(self.TASKS) - self.task_index
            hint = self._hint_for_task()

        return IncidentObservation(
            incident_id=inc.incident_id,
            title=inc.title,
            task_index=task_index,
            task_name=task_name,
            error_rate=inc.error_rate,
            latency_p99_ms=inc.latency_p99_ms,
            affected_users=inc.affected_users,
            logs=list(inc.logs),
            metrics=dict(inc.metrics),
            available_runbooks=list(inc.available_runbooks),
            history=list(self.history),
            remaining_tasks=remaining,
            hint=hint,
            done=self.done,
            session_id=self._session_id,
        )

    def _build_result(self, reward: float, info: Dict[str, Any]) -> StepResult:
        return StepResult(
            observation=self._build_observation(),
            reward=round(float(reward), 4),
            done=self.done,
            info=info,
            session_id=self._session_id,
        )

    def _hint_for_task(self) -> str:
        hints = {
            "classify_severity": (
                "Classify the incident as P0 (complete outage/revenue at risk), "
                "P1 (major degradation, many users affected), "
                "P2 (partial degradation, limited impact), or "
                "P3 (minor issue, no current user impact). "
                "Set the 'severity' field."
            ),
            "identify_root_cause": (
                "Look beyond the service showing errors — identify the underlying "
                "root cause service by analysing metrics and log patterns carefully. "
                "Set 'root_cause_service' and 'root_cause_reason' citing log evidence."
            ),
            "execute_response": (
                "Select the correct runbook_id from available_runbooks, estimate "
                "eta_minutes for resolution, and write a status_update for customers "
                "(≥30 words, do NOT mention internal service names like postgres/redis/k8s, "
                "do NOT promise a specific fix time)."
            ),
        }
        return hints.get(self.current_task_name, "Complete the current task.")


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _action_summary(task_name: str, action: IncidentAction) -> Dict[str, Any]:
    if task_name == "classify_severity":
        return {"severity": action.severity}
    if task_name == "identify_root_cause":
        return {"root_cause_service": action.root_cause_service,
                "root_cause_reason": action.root_cause_reason[:100]}
    return {"runbook_id": action.runbook_id,
            "eta_minutes": action.eta_minutes,
            "status_update": action.status_update[:120]}


def _extract_evidence_keywords(inc: IncidentData) -> List[str]:
    """
    Pull key terms that indicate awareness of the root cause.
    Includes: service name (with/without hyphens), service name parts,
    and log-level keywords from lines mentioning the root cause service.
    """
    svc = inc.true_root_cause_service.lower()
    keywords: List[str] = [svc, svc.replace("-", " ")]
    # Add individual meaningful parts (e.g. "postgres", "payments", "redis", "sessions")
    keywords.extend(p for p in svc.replace("-", " ").split() if len(p) > 3)
    # Add metric/log keywords from lines referencing the root cause
    for log_line in inc.logs:
        ll = log_line.lower()
        if svc in ll or any(p in ll for p in svc.split("-")):
            words = re.findall(r"[a-z0-9_\-]{4,}", ll)
            keywords.extend(words)
    return list(set(keywords))


def _infer_family(service: str) -> str:
    """Guess the service family from the name."""
    s = service.lower()
    if any(k in s for k in ["postgres", "mysql", "mongo", "database", "db", "rds"]):
        return "database"
    if any(k in s for k in ["redis", "memcached", "cache", "cdn"]):
        return "cache"
    if any(k in s for k in ["kafka", "rabbit", "queue", "pubsub", "sqs"]):
        return "service"
    if any(k in s for k in ["dns", "network", "nginx", "proxy", "gateway"]):
        return "network"
    if any(k in s for k in ["tls", "cert", "ssl", "config"]):
        return "config"
    if any(k in s for k in ["gpu", "cpu", "memory", "resource"]):
        return "resource"
    return "service"
