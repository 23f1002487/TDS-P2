"""
Visualization - Utilities for creating charts and visualizations
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from io import BytesIO
import base64
from .fallback_strategies import VisualizationFallbackStrategy, VizEngine

logger = logging.getLogger(__name__)


class Visualizer:
    """Handle various visualization tasks"""
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Check visualization capabilities
        self.availability = VisualizationFallbackStrategy.check_availability()
        self.selected_engine = VisualizationFallbackStrategy.select_engine(
            prefer_interactive=True,
            need_static_export=True,
            availability=self.availability
        )
    
    @staticmethod
    def create_bar_chart(data, x_col, y_col, title="Bar Chart", xlabel=None, ylabel=None):
        """Create a bar chart"""
        logger.info(f"Creating bar chart: {title}")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(data[x_col], data[y_col])
            ax.set_title(title)
            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            raise
        finally:
            plt.close()

    # ================== PLOTLY SUPPORT ==================
    @staticmethod
    def create_plotly_chart(data, chart_type: str, x_col: str, y_col: str, title: str = "Chart", registry=None):
        """Create an interactive Plotly chart. Falls back to Matplotlib if plotly unavailable.
        Returns dict with base64 image and optionally JSON spec.
        
        Args:
            data: DataFrame to visualize
            chart_type: Type of chart
            x_col: X-axis column
            y_col: Y-axis column
            title: Chart title
            registry: Optional CapabilityRegistry to record engine used
        """
        try:
            import plotly.express as px
            fig = None
            if chart_type == 'bar_chart':
                fig = px.bar(data, x=x_col, y=y_col, title=title)
            elif chart_type == 'line_chart':
                fig = px.line(data, x=x_col, y=y_col, title=title)
            elif chart_type == 'scatter_plot':
                fig = px.scatter(data, x=x_col, y=y_col, title=title)
            elif chart_type == 'pie_chart':
                fig = px.pie(data, names=x_col, values=y_col, title=title)
            elif chart_type == 'histogram':
                fig = px.histogram(data, x=x_col, title=title)
            else:
                fig = px.bar(data, x=x_col, y=y_col, title=title)

            # Export static image if kaleido present
            img_b64 = None
            kaleido_available = False
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                img_b64 = base64.b64encode(png_bytes).decode('utf-8')
                kaleido_available = True
            except Exception as e:
                logger.warning(f"Plotly static export failed: {e}")

            if registry:
                registry.record("viz_engine", "plotly")
                registry.record("viz_static_export", kaleido_available)

            return {
                'type': 'plotly',
                'chart_type': chart_type,
                'title': title,
                'image_base64': img_b64,
                'spec': fig.to_json()
            }
        except Exception as e:
            logger.warning(f"Plotly unavailable or failed ({e}); falling back to Matplotlib")
            if registry:
                registry.record("viz_engine", "matplotlib-fallback")
            # Fallback to existing bar chart
            b64 = Visualizer.create_bar_chart(data, x_col, y_col, title)
            return {
                'type': 'matplotlib-fallback',
                'chart_type': chart_type,
                'title': title,
                'image_base64': b64,
                'spec': None
            }
    
    @staticmethod
    def create_line_chart(data, x_col, y_col, title="Line Chart", xlabel=None, ylabel=None):
        """Create a line chart"""
        logger.info(f"Creating line chart: {title}")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data[x_col], data[y_col], marker='o')
            ax.set_title(title)
            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            raise
        finally:
            plt.close()
    
    @staticmethod
    def create_scatter_plot(data, x_col, y_col, title="Scatter Plot", xlabel=None, ylabel=None):
        """Create a scatter plot"""
        logger.info(f"Creating scatter plot: {title}")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data[x_col], data[y_col], alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            raise
        finally:
            plt.close()
    
    @staticmethod
    def create_pie_chart(data, labels_col, values_col, title="Pie Chart"):
        """Create a pie chart"""
        logger.info(f"Creating pie chart: {title}")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(data[values_col], labels=data[labels_col], autopct='%1.1f%%')
            ax.set_title(title)
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            raise
        finally:
            plt.close()
    
    @staticmethod
    def create_histogram(data, col, bins=30, title="Histogram", xlabel=None):
        """Create a histogram"""
        logger.info(f"Creating histogram: {title}")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data[col], bins=bins, edgecolor='black', alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel(xlabel or col)
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            raise
        finally:
            plt.close()
    
    @staticmethod
    def create_heatmap(data, title="Heatmap"):
        """Create a heatmap (typically for correlation matrix)"""
        logger.info(f"Creating heatmap: {title}")
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            
            return Visualizer._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            raise
        finally:
            plt.close()
    
    @staticmethod
    def _fig_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{img_base64}"
