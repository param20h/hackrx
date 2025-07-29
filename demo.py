"""
Demo script to showcase the Document Intelligence System capabilities.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import DocumentIntelligenceSystem
from utils import generate_test_queries, create_response_summary, format_currency
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_header():
    """Print a nice header for the demo."""
    header_text = Text("Document Intelligence System Demo", style="bold blue")
    console.print(Panel(header_text, expand=False))
    console.print()


def print_system_status(system):
    """Print system status information."""
    console.print("[bold]System Status:[/bold]")
    
    try:
        status = system.get_system_status()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("System", status["system_status"])
        
        if "components" in status:
            for component, comp_status in status["components"].items():
                table.add_row(component.replace("_", " ").title(), comp_status)
        
        if "collection_stats" in status:
            stats = status["collection_stats"]
            table.add_row("Indexed Documents", str(stats.get("total_chunks", 0)))
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error getting system status: {e}[/red]")


def demo_document_processing(system):
    """Demonstrate document processing capabilities."""
    console.print("[bold]Document Processing Demo:[/bold]")
    
    # Get sample documents from data directory
    docs_dir = Path(__file__).parent.parent / "data" / "documents"
    
    if docs_dir.exists():
        console.print(f"Adding documents from: {docs_dir}")
        
        try:
            statuses = system.add_documents_from_directory(str(docs_dir))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Document", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")
            
            for status in statuses:
                status_color = "green" if status.status == "completed" else "red"
                table.add_row(
                    status.document_id,
                    f"[{status_color}]{status.status}[/{status_color}]",
                    status.message or status.error or ""
                )
            
            console.print(table)
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error processing documents: {e}[/red]")
    else:
        console.print("[yellow]No documents directory found. Skipping document processing.[/yellow]")
        console.print()


def demo_query_processing(system):
    """Demonstrate query processing with various examples."""
    console.print("[bold]Query Processing Demo:[/bold]")
    
    # Get test queries
    test_queries = generate_test_queries()
    
    for i, query_data in enumerate(test_queries[:5], 1):  # Process first 5 queries
        console.print(f"\n[bold cyan]Query {i}:[/bold cyan] {query_data['query']}")
        console.print(f"[dim]{query_data['description']}[/dim]")
        
        try:
            # Process the query
            response = system.process_query(query_data['query'])
            
            # Create a results table
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Field", style="bold")
            table.add_column("Value")
            
            # Add decision information
            decision = response.decision
            decision_color = {
                "approved": "green",
                "rejected": "red", 
                "pending": "yellow",
                "requires_review": "orange"
            }.get(decision.decision_type.value, "white")
            
            table.add_row(
                "Decision:",
                f"[{decision_color}]{decision.decision_type.value.upper()}[/{decision_color}]"
            )
            
            if decision.amount:
                table.add_row(
                    "Amount:",
                    format_currency(decision.amount, decision.currency)
                )
            
            table.add_row(
                "Confidence:",
                f"{decision.confidence:.0%}"
            )
            
            table.add_row(
                "Processing Time:",
                f"{response.processing_time:.2f}s"
            )
            
            # Add extracted entities
            if response.structured_query.entities:
                entities_text = ", ".join([
                    f"{e.entity_type}: {e.value}" 
                    for e in response.structured_query.entities[:3]
                ])
                table.add_row("Key Entities:", entities_text)
            
            console.print(table)
            
            # Show reasoning (truncated)
            reasoning = decision.reasoning
            if len(reasoning) > 150:
                reasoning = reasoning[:150] + "..."
            console.print(f"[dim]Reasoning: {reasoning}[/dim]")
            
            # Show supporting clauses if any
            if response.supporting_clauses:
                console.print(f"[dim]Supporting clauses: {len(response.supporting_clauses)} found[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
        
        console.print("-" * 60)


def demo_api_examples(system):
    """Show API usage examples."""
    console.print("[bold]API Usage Examples:[/bold]")
    
    examples = [
        {
            "endpoint": "POST /query",
            "description": "Process a natural language query",
            "example": {
                "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
                "context": None
            }
        },
        {
            "endpoint": "POST /documents/upload",
            "description": "Upload documents for processing",
            "example": "Upload PDF, DOCX, or TXT files"
        },
        {
            "endpoint": "GET /status",
            "description": "Get system status and statistics",
            "example": "Returns system health and document count"
        }
    ]
    
    for example in examples:
        console.print(f"\n[bold cyan]{example['endpoint']}[/bold cyan]")
        console.print(f"{example['description']}")
        
        if isinstance(example['example'], dict):
            console.print(f"[dim]Example: {example['example']}[/dim]")
        else:
            console.print(f"[dim]{example['example']}[/dim]")
    
    console.print("\n[bold]To start the API server:[/bold]")
    console.print("python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000")
    console.print()


def main():
    """Run the complete demo."""
    print_header()
    
    console.print("Initializing Document Intelligence System...")
    console.print()
    
    try:
        # Initialize the system
        system = DocumentIntelligenceSystem()
        console.print("[green]✓ System initialized successfully![/green]")
        console.print()
        
        # Show system status
        print_system_status(system)
        
        # Demo document processing
        demo_document_processing(system)
        
        # Demo query processing
        demo_query_processing(system)
        
        # Show API examples
        demo_api_examples(system)
        
        console.print("[bold green]Demo completed successfully![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Set your OpenAI API key in .env file")
        console.print("2. Install required dependencies: pip install -r requirements.txt")
        console.print("3. Add your own documents to data/documents/")
        console.print("4. Start the API server for web interface")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("\nTroubleshooting:")
        console.print("1. Ensure all dependencies are installed")
        console.print("2. Check your .env configuration")
        console.print("3. Verify document paths are accessible")


if __name__ == "__main__":
    main()
