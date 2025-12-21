"""
Маршруты для мониторинга системы
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import datetime

from src.ml_pipeline.monitoring.drift_detection import DriftMonitor
from src.utils.logger import monitoring_logger, api_logger
from src.utils.database import get_database_manager
from src.api.middleware.auth import require_auth, require_role

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Модели Pydantic
class DriftAlert(BaseModel):
    """Алерт дрифта"""
    level: str
    type: str
    message: str
    details: Dict[str, Any]
    timestamp: str

class MonitoringStats(BaseModel):
    """Статистика мониторинга"""
    data_drift_score: float
    concept_drift_score: float
    drifted_columns: List[str]
    alerts_count: int
    last_check: str

class MonitoringRequest(BaseModel):
    """Запрос на запуск мониторинга"""
    hours: int = 24
    force: bool = False

# Инициализация монитора дрифта
drift_monitor = DriftMonitor()

@router.post("/drift/check")
@require_auth()
@require_role(["admin", "data_scientist", "monitoring"])
async def check_drift(
    request: MonitoringRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Запуск проверки дрифта"""
    try:
        monitoring_logger.info(
            f"Starting drift check for last {request.hours} hours"
        )
        
        # Запуск в фоновом режиме
        background_tasks.add_task(
            run_drift_monitoring,
            hours=request.hours,
            force=request.force
        )
        
        return {
            "status": "started",
            "hours": request.hours,
            "message": "Drift monitoring started in background",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        monitoring_logger.error(f"Failed to start drift check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def run_drift_monitoring(hours: int, force: bool = False):
    """Фоновая задача для мониторинга дрифта"""
    try:
        result = drift_monitor.monitor(hours=hours)
        
        monitoring_logger.info(
            f"Drift monitoring completed",
            extra={"result": result}
        )
        
        # Отправка алертов при необходимости
        if result.get("data_drift_detected") or result.get("concept_drift_detected"):
            await send_drift_alerts(result)
        
    except Exception as e:
        monitoring_logger.error(f"Drift monitoring failed: {e}", exc_info=True)

@router.get("/drift/results")
@require_auth()
@require_permission(["monitoring:read"])
async def get_drift_results(
    limit: int = 10,
    days: int = 7
) -> List[Dict[str, Any]]:
    """Получение результатов мониторинга дрифта"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = """
        SELECT 
            monitoring_type,
            results,
            created_at,
            success
        FROM monitoring_results
        WHERE created_at >= NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
        LIMIT %s
        """
        
        results = db_pool.execute_query(query, (days, limit))
        
        return [
            {
                "type": r["monitoring_type"],
                "results": r["results"],
                "timestamp": r["created_at"].isoformat(),
                "success": r["success"]
            }
            for r in results
        ]
        
    except Exception as e:
        monitoring_logger.error(f"Failed to get drift results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift/stats")
@require_auth()
@require_permission(["monitoring:read"])
async def get_drift_stats() -> MonitoringStats:
    """Получение статистики дрифта"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Последняя проверка дрифта
        query = """
        SELECT results
        FROM monitoring_results
        WHERE monitoring_type = 'data_drift'
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        result = db_pool.execute_query(query)
        
        if not result:
            return MonitoringStats(
                data_drift_score=0.0,
                concept_drift_score=0.0,
                drifted_columns=[],
                alerts_count=0,
                last_check=datetime.datetime.now().isoformat()
            )
        
        drift_results = result[0]["results"]
        
        # Подсчет алертов
        alerts_query = """
        SELECT COUNT(*) as alerts_count
        FROM drift_alerts
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
        
        alerts_result = db_pool.execute_query(alerts_query)[0]
        
        return MonitoringStats(
            data_drift_score=drift_results.get("data_drift_score", 0.0),
            concept_drift_score=drift_results.get("concept_drift_score", 0.0),
            drifted_columns=drift_results.get("drifted_columns", []),
            alerts_count=alerts_result["alerts_count"],
            last_check=result[0].get("created_at", datetime.datetime.now()).isoformat()
        )
        
    except Exception as e:
        monitoring_logger.error(f"Failed to get drift stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
@require_auth()
@require_role(["admin", "monitoring"])
async def get_alerts(
    level: Optional[str] = None,
    days: int = 7,
    limit: int = 100
) -> List[DriftAlert]:
    """Получение алертов"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = """
        SELECT 
            alert_level,
            alert_type,
            message,
            details,
            created_at
        FROM drift_alerts
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        
        if level:
            query += " AND alert_level = %s"
            params.append(level)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        alerts = db_pool.execute_query(query, tuple(params))
        
        return [
            DriftAlert(
                level=alert["alert_level"],
                type=alert["alert_type"],
                message=alert["message"],
                details=alert["details"],
                timestamp=alert["created_at"].isoformat()
            )
            for alert in alerts
        ]
        
    except Exception as e:
        monitoring_logger.error(f"Failed to get alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/acknowledge/{alert_id}")
@require_auth()
@require_role(["admin", "monitoring"])
async def acknowledge_alert(alert_id: int):
    """Подтверждение алерта"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        query = """
        UPDATE drift_alerts
        SET acknowledged = TRUE,
            acknowledged_at = NOW(),
            acknowledged_by = %s
        WHERE id = %s
        """
        
        db_pool.execute_query(
            query,
            ("api_user", alert_id),  # Заменить на реального пользователя
            fetch=False
        )
        
        monitoring_logger.info(f"Alert {alert_id} acknowledged")
        
        return {"message": f"Alert {alert_id} acknowledged"}
        
    except Exception as e:
        monitoring_logger.error(f"Failed to acknowledge alert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
@require_auth()
@require_permission(["monitoring:read"])
async def get_performance_metrics(
    interval: str = "1h",
    metric: str = "latency"
) -> Dict[str, Any]:
    """Получение метрик производительности"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Определение интервала для GROUP BY
        interval_map = {
            "1h": "date_trunc('hour', created_at)",
            "1d": "date_trunc('day', created_at)",
            "1w": "date_trunc('week', created_at)"
        }
        
        trunc_expr = interval_map.get(interval, "date_trunc('hour', created_at)")
        
        query = f"""
        SELECT 
            {trunc_expr} as time_interval,
            COUNT(*) as request_count,
            AVG(processing_time_ms) as avg_latency,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_latency,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY processing_time_ms) as p99_latency,
            SUM(CASE WHEN (prediction_result->>'class')::int = 1 THEN 1 ELSE 0 END) as bad_predictions
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '7 days'
        GROUP BY time_interval
        ORDER BY time_interval
        """
        
        metrics = db_pool.execute_query(query)
        
        # Форматирование результата
        formatted_metrics = []
        for m in metrics:
            formatted_metrics.append({
                "timestamp": m["time_interval"].isoformat(),
                "request_count": m["request_count"],
                "avg_latency_ms": float(m["avg_latency"]),
                "p95_latency_ms": float(m["p95_latency"]),
                "p99_latency_ms": float(m["p99_latency"]),
                "bad_predictions": m["bad_predictions"],
                "success_rate": (m["request_count"] - m["bad_predictions"]) / m["request_count"] * 100
            })
        
        return {
            "interval": interval,
            "metric": metric,
            "metrics": formatted_metrics
        }
        
    except Exception as e:
        monitoring_logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system")
@require_auth()
@require_role(["admin", "monitoring"])
async def get_system_metrics():
    """Получение системных метрик"""
    import psutil
    
    try:
        # Системные метрики
        system_metrics = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "load_avg": psutil.getloadavg()
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / 1024**3,
                "available_gb": psutil.virtual_memory().available / 1024**3,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / 1024**3,
                "used_gb": psutil.disk_usage('/').used / 1024**3,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
        
        # Метрики процесса
        process = psutil.Process()
        process_metrics = {
            "pid": process.pid,
            "memory_mb": process.memory_info().rss / 1024**2,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "connections": len(process.connections())
        }
        
        return {
            "system": system_metrics,
            "process": process_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        monitoring_logger.error(f"Failed to get system metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain")
@require_auth()
@require_role(["admin", "ml_engineer"])
async def trigger_retraining():
    """Запуск переобучения модели"""
    try:
        monitoring_logger.info("Manual retraining triggered via API")
        
        # Проверка условий для переобучения
        drift_stats = await get_drift_stats()
        
        if (drift_stats.data_drift_score < 0.3 and 
            drift_stats.concept_drift_score < 0.3):
            return {
                "status": "skipped",
                "reason": "Drift scores below threshold",
                "data_drift_score": drift_stats.data_drift_score,
                "concept_drift_score": drift_stats.concept_drift_score
            }
        
        # Запуск переобучения в фоне
        background_tasks = BackgroundTasks()
        background_tasks.add_task(run_retraining_pipeline)
        
        return {
            "status": "started",
            "message": "Retraining pipeline started in background",
            "drift_scores": {
                "data": drift_stats.data_drift_score,
                "concept": drift_stats.concept_drift_score
            }
        }
        
    except Exception as e:
        monitoring_logger.error(f"Failed to trigger retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def run_retraining_pipeline():
    """Фоновая задача для переобучения модели"""
    try:
        monitoring_logger.info("Starting retraining pipeline")
        
        # Здесь можно вызвать скрипт переобучения
        # Например: os.system("python src/ml_pipeline/training/train_model.py")
        
        monitoring_logger.info("Retraining pipeline completed")
        
    except Exception as e:
        monitoring_logger.error(f"Retraining pipeline failed: {e}", exc_info=True)

async def send_drift_alerts(drift_result: Dict[str, Any]):
    """Отправка алертов о дрифте"""
    try:
        db_manager = get_database_manager()
        db_pool = db_manager.get_pool("default")
        
        # Создание алертов на основе результатов
        alerts = []
        
        if drift_result.get("data_drift_detected"):
            alerts.append({
                "alert_level": "WARNING",
                "alert_type": "DATA_DRIFT",
                "message": f"Data drift detected: score={drift_result['data_drift_score']:.3f}",
                "details": {
                    "score": drift_result["data_drift_score"],
                    "drifted_columns": drift_result.get("drifted_columns", []),
                    "threshold": 0.3
                }
            })
        
        if drift_result.get("concept_drift_detected"):
            alerts.append({
                "alert_level": "CRITICAL",
                "alert_type": "CONCEPT_DRIFT",
                "message": f"Concept drift detected: score={drift_result['concept_drift_score']:.3f}",
                "details": {
                    "score": drift_result["concept_drift_score"],
                    "threshold": 0.3
                }
            })
        
        # Сохранение алертов в БД
        for alert in alerts:
            query = """
            INSERT INTO drift_alerts (
                alert_level,
                alert_type,
                message,
                details,
                created_at
            ) VALUES (%s, %s, %s, %s, NOW())
            """
            
            db_pool.execute_query(
                query,
                (
                    alert["alert_level"],
                    alert["alert_type"],
                    alert["message"],
                    alert["details"]
                ),
                fetch=False
            )
        
        # Отправка уведомлений (email, Slack, etc.)
        if alerts:
            await send_notifications(alerts)
        
        monitoring_logger.info(f"Created {len(alerts)} drift alerts")
        
    except Exception as e:
        monitoring_logger.error(f"Failed to send drift alerts: {e}", exc_info=True)

async def send_notifications(alerts: List[Dict[str, Any]]):
    """Отправка уведомлений"""
    # Здесь можно реализовать отправку через:
    # - Email
    # - Slack
    # - Telegram
    # - PagerDuty
    pass