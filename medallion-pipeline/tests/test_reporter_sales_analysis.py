from pathlib import Path

from agents.reporter import create_report_insight


def test_create_report_insight_builds_yearly_product_and_location_report(tmp_path):
    landing_dir = Path("data/landing")
    source_files = [
        str(landing_dir / "sales_data.csv"),
        str(landing_dir / "products.csv"),
        str(landing_dir / "stores.csv"),
    ]

    result = create_report_insight(
        "gold/sample.parquet",
        report_path=tmp_path,
        business_question="What is yearly sales by product and location?",
        source_files=source_files,
    )

    analysis = result["analysis"]
    assert analysis["yearly_sales"]
    assert analysis["yearly_sales_by_product"]
    assert analysis["yearly_sales_by_location"]


def test_create_report_insight_returns_question_specific_analysis(tmp_path):
    landing_dir = Path("data/landing")
    source_files = [
        str(landing_dir / "sales_data.csv"),
        str(landing_dir / "products.csv"),
        str(landing_dir / "stores.csv"),
    ]

    result = create_report_insight(
        "gold/sample.parquet",
        report_path=tmp_path,
        business_question="What are yearly sales by product?",
        source_files=source_files,
    )

    analysis = result["analysis"]
    assert analysis["yearly_sales_by_product"]
    assert not analysis.get("yearly_sales_by_location")
    assert not analysis.get("monthly_profit")
    assert (tmp_path / "report_insights.pdf").exists()
