from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str = Field(default="", description="Unique identifier for this pipeline run")
    status: str = Field(default="initialized", description="Current status of the pipeline")

    uploaded_files: list[str] = Field(
        default_factory=list,
        description="Paths to uploaded CSV files",
    )
    business_intent: str = Field(
        default="",
        description="User's question in plain English",
    )

    profile_path: str = Field(
        default="",
        description="Path to the data profile JSON file",
    )

    sttm_bronze_path: str = Field(
        default="",
        description="Path to Bronze layer transformation rules",
    )
    sttm_silver_path: str = Field(
        default="",
        description="Path to Silver layer transformation rules",
    )
    sttm_gold_path: str = Field(
        default="",
        description="Path to Gold layer transformation rules",
    )
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human has approved the transformation rules",
    )

    bronze_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Bronze Parquet output files",
    )
    silver_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Silver Parquet output files",
    )
    gold_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Gold Parquet output files",
    )

    report_path: str = Field(
        default="",
        description="Path to the final HTML report",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message when the pipeline fails",
    )
