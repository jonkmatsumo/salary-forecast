"""MCP tool handlers that map to service layer."""

from typing import Any, Dict, cast

from src.api.dto.models import ModelMetadata
from src.services.analytics_service import AnalyticsService
from src.services.inference_service import InferenceService
from src.services.model_registry import ModelRegistry
from src.services.training_service import TrainingService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPToolHandler:
    """Handler for MCP tool invocations."""

    def __init__(self):
        """Initialize handler. Returns: None."""
        self.inference_service = InferenceService()
        self.training_service = TrainingService()
        self.workflow_service = None
        self.model_registry = ModelRegistry()
        self.analytics_service = AnalyticsService()

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call. Args: tool_name (str): Tool name. arguments (Dict[str, Any]): Tool arguments. Returns: Dict[str, Any]: Tool result. Raises: ValueError: If tool not found or invalid arguments."""
        handler_map = {
            "list_models": self._handle_list_models,
            "get_model_details": self._handle_get_model_details,
            "get_model_schema": self._handle_get_model_schema,
            "predict_salary": self._handle_predict_salary,
            "start_training": self._handle_start_training,
            "get_training_status": self._handle_get_training_status,
            "start_configuration_workflow": self._handle_start_configuration_workflow,
            "confirm_classification": self._handle_confirm_classification,
            "confirm_encoding": self._handle_confirm_encoding,
            "finalize_configuration": self._handle_finalize_configuration,
            "get_feature_importance": self._handle_get_feature_importance,
        }

        handler = handler_map.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")

        return await handler(arguments)

    async def _handle_list_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_models tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        limit = args.get("limit", 50)
        offset = args.get("offset", 0)
        experiment_name = args.get("experiment_name")

        runs = self.model_registry.list_models()

        if experiment_name:
            runs = [r for r in runs if r.get("tags.experiment_name") == experiment_name]

        total = len(runs)
        paginated_runs = runs[offset : offset + limit]

        models = [
            ModelMetadata(
                run_id=run["run_id"],
                start_time=run["start_time"],
                model_type=run.get("tags.model_type", "XGBoost"),
                cv_mean_score=run.get("metrics.cv_mean_score"),
                dataset_name=run.get("tags.dataset_name", "Unknown"),
                additional_tag=run.get("tags.additional_tag"),
            )
            for run in paginated_runs
        ]

        return {
            "models": [m.model_dump() for m in models],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _handle_get_model_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_model_details tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.dto.models import (
            ModelDetailsResponse,
            ModelMetadata,
            ModelSchema,
            ProximityFeatureSchema,
            RankedFeatureSchema,
        )
        from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
        from src.services.inference_service import ModelNotFoundError

        run_id = args["run_id"]

        try:
            model = self.inference_service.load_model(run_id)
            schema = self.inference_service.get_model_schema(model)

            ranked_features = [
                RankedFeatureSchema(
                    name=name,
                    levels=list(model.ranked_encoders[name].mapping.keys()),
                    encoding_type="ranked",
                )
                for name in schema.ranked_features
            ]

            proximity_features = [
                ProximityFeatureSchema(name=name, encoding_type="proximity")
                for name in schema.proximity_features
            ]

            model_schema = ModelSchema(
                ranked_features=ranked_features,
                proximity_features=proximity_features,
                numerical_features=schema.numerical_features,
            )

            runs = self.model_registry.list_models()
            run_data = next((r for r in runs if r["run_id"] == run_id), None)

            if not run_data:
                raise APIModelNotFoundError(run_id)

            metadata = ModelMetadata(
                run_id=run_id,
                start_time=run_data["start_time"],
                model_type=run_data.get("tags.model_type", "XGBoost"),
                cv_mean_score=run_data.get("metrics.cv_mean_score"),
                dataset_name=run_data.get("tags.dataset_name", "Unknown"),
                additional_tag=run_data.get("tags.additional_tag"),
            )

            response = ModelDetailsResponse(
                run_id=run_id,
                metadata=metadata,
                model_schema=model_schema,
                feature_names=schema.all_feature_names,
                targets=schema.targets,
                quantiles=schema.quantiles,
            )
            return cast(Dict[str, Any], response.model_dump())
        except ModelNotFoundError as e:
            raise APIModelNotFoundError(run_id) from e

    async def _handle_get_model_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_model_schema tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.dto.models import (
            ModelSchema,
            ModelSchemaResponse,
            ProximityFeatureSchema,
            RankedFeatureSchema,
        )
        from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
        from src.services.inference_service import ModelNotFoundError

        run_id = args["run_id"]

        try:
            model = self.inference_service.load_model(run_id)
            schema = self.inference_service.get_model_schema(model)

            ranked_features = [
                RankedFeatureSchema(
                    name=name,
                    levels=list(model.ranked_encoders[name].mapping.keys()),
                    encoding_type="ranked",
                )
                for name in schema.ranked_features
            ]

            proximity_features = [
                ProximityFeatureSchema(name=name, encoding_type="proximity")
                for name in schema.proximity_features
            ]

            model_schema = ModelSchema(
                ranked_features=ranked_features,
                proximity_features=proximity_features,
                numerical_features=schema.numerical_features,
            )

            response = ModelSchemaResponse(run_id=run_id, model_schema=model_schema)
            return cast(Dict[str, Any], response.model_dump())
        except ModelNotFoundError as e:
            raise APIModelNotFoundError(run_id) from e

    async def _handle_predict_salary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle predict_salary tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.dto.inference import PredictionMetadata, PredictionResponse
        from src.api.exceptions import InvalidInputError as APIInvalidInputError
        from src.services.inference_service import InvalidInputError, ModelNotFoundError

        run_id = args["run_id"]
        features = args["features"]

        try:
            model = self.inference_service.load_model(run_id)
            result = self.inference_service.predict(model, features)

            from datetime import datetime

            metadata_dict = result.metadata if isinstance(result.metadata, dict) else {}
            metadata_obj = PredictionMetadata(
                model_run_id=metadata_dict.get("model_run_id", run_id),
                prediction_timestamp=metadata_dict.get("prediction_timestamp", datetime.now()),
                location_zone=metadata_dict.get("location_zone"),
            )
            response = PredictionResponse(
                predictions=result.predictions,
                metadata=metadata_obj,
            )
            return cast(Dict[str, Any], response.model_dump())
        except ModelNotFoundError as e:
            from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError

            raise APIModelNotFoundError(run_id) from e
        except InvalidInputError as e:
            raise APIInvalidInputError(str(e)) from e

    async def _handle_start_training(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start_training tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.dto.training import TrainingJobRequest
        from src.api.routers.training import start_training
        from src.api.storage import DatasetStorage

        dataset_id = args["dataset_id"]
        config = args["config"]
        remove_outliers = args.get("remove_outliers", True)
        do_tune = args.get("do_tune", False)
        n_trials = args.get("n_trials")
        additional_tag = args.get("additional_tag")
        dataset_name = args.get("dataset_name")

        storage = DatasetStorage()
        dataset = storage.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        request = TrainingJobRequest(
            dataset_id=dataset_id,
            config=config,
            remove_outliers=remove_outliers,
            do_tune=do_tune,
            n_trials=n_trials,
            additional_tag=additional_tag,
            dataset_name=dataset_name,
        )

        response = await start_training(
            request=request,
            user="mcp",
            training_service=self.training_service,
        )
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_get_training_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_training_status tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.routers.training import get_training_job_status

        job_id = args["job_id"]
        response = await get_training_job_status(
            job_id, user="mcp", training_service=self.training_service
        )
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_start_configuration_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start_configuration_workflow tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from io import StringIO

        import pandas as pd

        from src.api.routers.workflow import start_workflow

        data = args["data"]
        columns = args["columns"]
        dtypes = args["dtypes"]
        dataset_size = args["dataset_size"]
        provider = args.get("provider", "openai")
        preset = args.get("preset")

        pd.read_json(StringIO(data), orient="records")

        request_body = {
            "data": data,
            "columns": columns,
            "dtypes": dtypes,
            "dataset_size": dataset_size,
            "provider": provider,
        }
        if preset:
            request_body["preset"] = preset

        response = await start_workflow(request_body, user="mcp")
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_confirm_classification(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle confirm_classification tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.routers.workflow import confirm_classification

        workflow_id = args["workflow_id"]
        modifications = args["modifications"]

        from src.api.dto.workflow import (
            ClassificationConfirmationRequest,
            ClassificationModifications,
        )

        request_body = ClassificationConfirmationRequest(
            modifications=ClassificationModifications(
                targets=modifications.get("targets", []),
                features=modifications.get("features", []),
                ignore=modifications.get("ignore", []),
            )
        )

        response = await confirm_classification(workflow_id, request_body, user="mcp")
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_confirm_encoding(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle confirm_encoding tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.routers.workflow import confirm_encoding

        workflow_id = args["workflow_id"]
        modifications = args["modifications"]

        from src.api.dto.workflow import EncodingConfirmationRequest, EncodingModifications

        encodings = {}
        for col, enc_config in modifications.get("encodings", {}).items():
            from src.api.dto.workflow import EncodingConfig

            encodings[col] = EncodingConfig(
                type=enc_config.get("type"),
                mapping=enc_config.get("mapping"),
                reasoning=enc_config.get("reasoning"),
            )

        optional_encodings = {}
        for col, opt_enc_config in modifications.get("optional_encodings", {}).items():
            from src.api.dto.workflow import OptionalEncodingConfig

            optional_encodings[col] = OptionalEncodingConfig(
                type=opt_enc_config.get("type"),
                params=opt_enc_config.get("params", {}),
            )

        request_body = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings=encodings,
                optional_encodings=optional_encodings,
            )
        )

        response = await confirm_encoding(workflow_id, request_body, user="mcp")
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_finalize_configuration(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle finalize_configuration tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.routers.workflow import finalize_configuration

        workflow_id = args["workflow_id"]
        config_updates = args["config_updates"]

        from src.api.dto.workflow import (
            ConfigurationFinalizationRequest,
            FeatureConfig,
            Hyperparameters,
        )

        features = [
            FeatureConfig(name=f["name"], monotone_constraint=f["monotone_constraint"])
            for f in config_updates.get("features", [])
        ]

        request_body = ConfigurationFinalizationRequest(
            features=features,
            quantiles=config_updates.get("quantiles", []),
            hyperparameters=Hyperparameters(
                training=config_updates.get("hyperparameters", {}).get("training", {}),
                cv=config_updates.get("hyperparameters", {}).get("cv", {}),
            ),
            location_settings=config_updates.get("location_settings"),
        )

        response = await finalize_configuration(workflow_id, request_body, user="mcp")
        return cast(Dict[str, Any], response.model_dump())

    async def _handle_get_feature_importance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_feature_importance tool. Args: args (Dict[str, Any]): Arguments. Returns: Dict[str, Any]: Result."""
        from src.api.routers.analytics import get_feature_importance

        run_id = args["run_id"]
        target = args["target"]
        quantile = args["quantile"]

        response = await get_feature_importance(
            run_id,
            target,
            quantile,
            user="mcp",
            inference_service=self.inference_service,
            analytics_service=self.analytics_service,
        )
        return cast(Dict[str, Any], response.model_dump())
