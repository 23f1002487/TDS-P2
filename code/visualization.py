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

logger = logging.getLogger(__name__)


class Visualizer:
    """Handle various visualization tasks"""
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
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
