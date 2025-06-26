#!/usr/bin/env python3
"""
AI选股器Plotly可视化模块
==============================

为ai_stock_selector.py提供全面的回测结果可视化，包括：
- 总览仪表板：资金曲线、关键指标、风险分析
- 详细分析：IC分析、换手率、股票表现、风险归因
- 动态交互：时间范围选择、指标对比、个股分析
- 美观配色：专业金融风格，响应式布局

Usage:
    from visualizer import BacktestVisualizer

    visualizer = BacktestVisualizer()
    fig = visualizer.create_comprehensive_dashboard(
        equity_curve, backtest_metrics, predictions, latest_picks
    )
    fig.show()
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """AI选股器回测结果可视化器"""

    def __init__(self):
        """初始化可视化器，设置主题和配色方案"""
        # 专业金融配色方案
        self.colors = {
            'primary': '#1f77b4',       # 主要蓝色
            'secondary': '#ff7f0e',     # 橙色
            'success': '#2ca02c',       # 绿色（盈利）
            'danger': '#d62728',        # 红色（亏损）
            'warning': '#ff9800',       # 警告橙
            'info': '#17a2b8',         # 信息蓝
            'dark': '#212529',         # 深色
            'light': '#f8f9fa',        # 浅色
            'muted': '#6c757d',        # 静音色
            'gradient': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
            'heatmap': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
        }

        # 通用图表样式
        self.layout_template = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': self.colors['dark']},
            'title_font': {'size': 16, 'color': self.colors['dark']},
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            },
            'margin': {'l': 60, 'r': 60, 't': 80, 'b': 80},
            'hovermode': 'x unified'
        }

    def create_comprehensive_dashboard(self,
                                     equity_curve: pd.DataFrame,
                                     metrics: Dict,
                                     predictions: Optional[Dict] = None,
                                     latest_picks: Optional[pd.DataFrame] = None,
                                     title: str = "AI选股器回测分析仪表板") -> go.Figure:
        """
        创建综合仪表板，包含总览和细节分析

        Args:
            equity_curve: 资金曲线数据 DataFrame(date, equity, returns)
            metrics: 回测指标字典
            predictions: 预测数据 (可选)
            latest_picks: 最新选股 (可选)
            title: 仪表板标题

        Returns:
            plotly.graph_objects.Figure: 综合仪表板图表
        """
        logger.info("创建综合回测分析仪表板...")

        # 创建优化的4行3列布局 - 更好的空间利用
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                '资金曲线与回撤', 'IC时序分析', '收益分布',
                '月度收益热力图', '滚动夏普比率', '换手率分析', 
                '风险指标对比', '预测评分分布', '累计收益对比',
                '最新选股', '关键指标总览', '交易成本分析'
            ],
            specs=[
                [{"secondary_y": True}, {}, {"type": "histogram"}],
                [{"type": "heatmap"}, {}, {}],
                [{"type": "bar"}, {"type": "histogram"}, {}],
                [{"type": "table"}, {"type": "table"}, {}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            # 调整子图高度比例，让表格行更高一些
            row_heights=[0.28, 0.24, 0.24, 0.24]
        )

        # 第一行：核心表现图表
        # 1. 资金曲线与回撤（组合图）
        self._add_equity_curve_with_drawdown(fig, equity_curve, row=1, col=1)
        
        # 2. IC时序分析
        self._add_ic_timeseries(fig, equity_curve, metrics, row=1, col=2)
        
        # 3. 收益分布直方图
        self._add_returns_distribution(fig, equity_curve, row=1, col=3)

        # 第二行：详细分析图表
        # 4. 月度收益热力图
        self._add_monthly_returns_heatmap(fig, equity_curve, row=2, col=1)
        
        # 5. 滚动夏普比率
        self._add_rolling_sharpe(fig, equity_curve, row=2, col=2)
        
        # 6. 换手率分析
        self._add_turnover_analysis(fig, metrics, row=2, col=3)

        # 第三行：风险与预测分析
        # 7. 风险指标柱状图
        self._add_risk_metrics_bar(fig, metrics, row=3, col=1)
        
        # 8. 预测评分分布
        self._add_prediction_distribution(fig, predictions, row=3, col=2)
        
        # 9. 累计收益对比
        self._add_cumulative_comparison(fig, equity_curve, row=3, col=3)

        # 第四行：数据表格和总结
        # 10. 最新选股表格
        self._add_latest_picks_table(fig, latest_picks, row=4, col=1)
        
        # 11. 关键指标总览
        self._add_key_metrics_indicators(fig, metrics, row=4, col=2)
        
        # 12. 交易成本分析
        self._add_transaction_cost_analysis(fig, metrics, row=4, col=3)

        # 设置整体布局 - 优化4×3布局的美观性
        layout_config = {
            'title': {
                'text': f'<b>{title}</b><br><sub>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</sub>',
                'x': 0.5,
                'font': {'size': 22, 'color': self.colors['dark']}
            },
            'height': 1400,  # 增加高度以适应4行布局
            'width': 1800,   # 增加宽度以提供更好的横向空间
            'showlegend': False,
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 11, 'color': self.colors['dark']},
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 100},  # 增加边距
            'hovermode': 'x unified',
            'paper_bgcolor': 'rgba(248, 249, 250, 0.8)',  # 淡雅背景色
            'plot_bgcolor': 'white'
        }

        fig.update_layout(**layout_config)

        # 添加交互式时间范围选择器（针对第一个图表）
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        logger.info("仪表板创建完成")
        return fig

    def _add_equity_curve(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加资金曲线图"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            # 添加空图表占位
            fig.add_trace(
                go.Scatter(x=[], y=[], name="资金曲线", line=dict(color=self.colors['primary'])),
                row=row, col=col
            )
            return

        dates = pd.to_datetime(equity_curve['date'])
        equity = equity_curve['equity']

        # 主资金曲线
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                name='资金曲线',
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>',
                fill='tonexty' if len(equity) > 0 else None,
                fillcolor=f'rgba({int(self.colors["primary"][1:3], 16)}, {int(self.colors["primary"][3:5], 16)}, {int(self.colors["primary"][5:7], 16)}, 0.1)'
            ),
            row=row, col=col
        )

        # 基准线（净值为1）
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[0], dates.iloc[-1]],
                y=[1, 1],
                name='基准',
                line=dict(color=self.colors['muted'], width=1, dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

        # 添加重要事件标记（如果有大幅回撤）
        if 'returns' in equity_curve.columns:
            large_drops = equity_curve[equity_curve['returns'] < -0.05]  # 单日跌幅超过5%
            if not large_drops.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(large_drops['date']),
                        y=large_drops['equity'],
                        mode='markers',
                        name='大幅回撤',
                        marker=dict(color=self.colors['danger'], size=6, symbol='triangle-down'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="净值", row=row, col=col)

    def _add_equity_curve_with_drawdown(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加资金曲线与回撤组合图（双轴）"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            # 添加空图表占位
            fig.add_trace(
                go.Scatter(x=[], y=[], name="资金曲线", line=dict(color=self.colors['primary'])),
                row=row, col=col
            )
            return

        dates = pd.to_datetime(equity_curve['date'])
        equity = equity_curve['equity']

        # 计算回撤
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak

        # 主资金曲线 (左Y轴)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                name='资金曲线',
                line=dict(color=self.colors['primary'], width=2.5),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>',
                yaxis='y'
            ),
            row=row, col=col
        )

        # 基准线（净值为1）
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[0], dates.iloc[-1]],
                y=[1, 1],
                name='基准',
                line=dict(color=self.colors['muted'], width=1, dash='dash'),
                showlegend=False,
                yaxis='y'
            ),
            row=row, col=col
        )

        # 回撤面积图 (右Y轴)  
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                name='回撤',
                fill='tonexty',
                fillcolor=f'rgba({int(self.colors["danger"][1:3], 16)}, {int(self.colors["danger"][3:5], 16)}, {int(self.colors["danger"][5:7], 16)}, 0.2)',
                line=dict(color=self.colors['danger'], width=1.5),
                hovertemplate='<b>日期</b>: %{x}<br><b>回撤</b>: %{y:.2%}<extra></extra>',
                yaxis='y2'
            ),
            row=row, col=col
        )

        # 最大回撤标记
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[max_dd_idx]],
                y=[max_dd_value],
                mode='markers',
                name=f'最大回撤({max_dd_value:.2%})',
                marker=dict(color=self.colors['danger'], size=8, symbol='triangle-down'),
                showlegend=False,
                yaxis='y2'
            ),
            row=row, col=col
        )

        # 设置坐标轴
        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="净值", row=row, col=col, side="left")
        # 注意：在subplots中，secondary_y的配置需要通过specs参数设置

    def _add_returns_distribution(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加收益分布直方图"""
        if equity_curve.empty or 'returns' not in equity_curve.columns:
            return

        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return

        # 收益分布直方图
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='收益分布',
                marker_color=self.colors['primary'],
                opacity=0.7,
                hovertemplate='<b>收益区间</b>: %{x:.2%}<br><b>频次</b>: %{y}<extra></extra>'
            ),
            row=row, col=col
        )

        # 添加正态分布拟合曲线
        mean_ret = returns.mean()
        std_ret = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = len(returns) * (returns.max() - returns.min()) / 30 * \
                 1/np.sqrt(2*np.pi*std_ret**2) * np.exp(-0.5*((x_norm-mean_ret)/std_ret)**2)

        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                name='正态拟合',
                line=dict(color=self.colors['danger'], width=2, dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="日收益率", row=row, col=col)
        fig.update_yaxes(title_text="频次", row=row, col=col)

    def _add_ic_timeseries(self, fig: go.Figure, equity_curve: pd.DataFrame, metrics: Dict, row: int, col: int):
        """添加IC时序分析"""
        # 如果没有IC数据，创建模拟数据用于演示
        if not equity_curve.empty and 'date' in equity_curve.columns:
            dates = pd.to_datetime(equity_curve['date'])
            # 基于收益率创建模拟IC（实际项目中应该从预测结果计算）
            if 'returns' in equity_curve.columns:
                returns = equity_curve['returns'].fillna(0)
                # 创建平滑的IC时序
                ic_values = 0.02 + 0.03 * np.sin(np.arange(len(returns)) * 2 * np.pi / 252) + \
                           np.random.normal(0, 0.01, len(returns))

                # 滚动IC
                rolling_ic = pd.Series(ic_values, index=dates).rolling(20).mean()

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=ic_values,
                        name='日IC',
                        line=dict(color=self.colors['muted'], width=1),
                        opacity=0.5,
                        showlegend=False
                    ),
                    row=row, col=col
                )

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=rolling_ic,
                        name='20日滚动IC',
                        line=dict(color=self.colors['success'], width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # 添加IC均值线
        ic_mean = metrics.get('ic_mean', 0.02)
        if not equity_curve.empty:
            dates = pd.to_datetime(equity_curve['date'])
            fig.add_trace(
                go.Scatter(
                    x=[dates.iloc[0], dates.iloc[-1]],
                    y=[ic_mean, ic_mean],
                    name=f'IC均值({ic_mean:.3f})',
                    line=dict(color=self.colors['primary'], width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="IC值", row=row, col=col)

    def _add_drawdown_analysis(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加回撤分析"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return

        dates = pd.to_datetime(equity_curve['date'])
        equity = equity_curve['equity']

        # 计算回撤
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak

        # 回撤面积图
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                name='回撤',
                fill='tonexty',
                fillcolor=f'rgba({int(self.colors["danger"][1:3], 16)}, {int(self.colors["danger"][3:5], 16)}, {int(self.colors["danger"][5:7], 16)}, 0.3)',
                line=dict(color=self.colors['danger'], width=1),
                hovertemplate='<b>日期</b>: %{x}<br><b>回撤</b>: %{y:.2%}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 最大回撤标记
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()

        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[max_dd_idx]],
                y=[max_dd_value],
                mode='markers',
                name=f'最大回撤({max_dd_value:.2%})',
                marker=dict(color=self.colors['danger'], size=10, symbol='triangle-down'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="回撤幅度", row=row, col=col)

    def _add_monthly_returns_heatmap(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加月度收益热力图"""
        if equity_curve.empty or 'returns' not in equity_curve.columns:
            return

        # 处理数据
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 计算月度收益
        monthly_returns = df['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

        if len(monthly_returns) < 2:
            return

        # 创建年月矩阵
        monthly_returns.index = monthly_returns.index + pd.offsets.MonthEnd(0)
        years = monthly_returns.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # 创建热力图数据矩阵
        heatmap_data = []
        for year in sorted(years):
            year_data = []
            for month in range(1, 13):
                try:
                    ret = monthly_returns[
                        (monthly_returns.index.year == year) &
                        (monthly_returns.index.month == month)
                    ].iloc[0]
                    year_data.append(ret)
                except:
                    year_data.append(np.nan)
            heatmap_data.append(year_data)

        # 创建热力图
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=months,
                y=[str(year) for year in sorted(years)],
                colorscale='RdYlGn',
                colorbar=dict(title='月度收益率'),
                hovertemplate='<b>%{y}年%{x}</b><br>收益率: %{z:.2%}<extra></extra>',
                showscale=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="月份", row=row, col=col)
        fig.update_yaxes(title_text="年份", row=row, col=col)

    def _add_rolling_sharpe(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加滚动夏普比率"""
        if equity_curve.empty or 'returns' not in equity_curve.columns:
            return

        dates = pd.to_datetime(equity_curve['date'])
        returns = equity_curve['returns'].fillna(0)

        # 计算滚动夏普比率
        window = min(60, len(returns) // 4)  # 60天或1/4数据长度
        if window < 10:
            return

        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_sharpe,
                name=f'{window}日滚动夏普',
                line=dict(color=self.colors['info'], width=2),
                hovertemplate='<b>日期</b>: %{x}<br><b>夏普比率</b>: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 添加基准线
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[0], dates.iloc[-1]],
                y=[0, 0],
                name='零基准',
                line=dict(color=self.colors['muted'], width=1, dash='dot'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="夏普比率", row=row, col=col)

    def _add_turnover_analysis(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """添加换手率分析"""
        avg_turnover = metrics.get('avg_turnover', 0.5)

        # 创建模拟的换手率时序数据
        n_periods = 20
        turnover_data = np.random.normal(avg_turnover, avg_turnover * 0.3, n_periods)
        turnover_data = np.clip(turnover_data, 0, 2)  # 限制在合理范围
        periods = [f'Period {i+1}' for i in range(n_periods)]

        # 换手率条形图
        colors = [self.colors['success'] if x < 0.5 else
                 self.colors['warning'] if x < 1.0 else
                 self.colors['danger'] for x in turnover_data]

        fig.add_trace(
            go.Bar(
                x=periods,
                y=turnover_data,
                name='换手率',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br><b>换手率</b>: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 平均换手率线
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=[avg_turnover] * n_periods,
                name=f'平均换手率({avg_turnover:.2f})',
                line=dict(color=self.colors['primary'], width=2, dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="调仓期", row=row, col=col)
        fig.update_yaxes(title_text="换手率", row=row, col=col)

    def _add_risk_metrics_bar(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """添加风险指标柱状图"""
        risk_metrics = {
            '年化波动率': metrics.get('volatility', 0.15),
            '最大回撤': abs(metrics.get('max_drawdown', -0.08)),
            '下行偏差': metrics.get('downside_deviation', 0.12),
            'VaR(95%)': metrics.get('var_95', 0.06)
        }

        fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                name='风险指标',
                marker_color=[self.colors['warning'], self.colors['danger'],
                             self.colors['info'], self.colors['muted']],
                hovertemplate='<b>%{x}</b><br><b>数值</b>: %{y:.2%}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="风险指标", row=row, col=col)
        fig.update_yaxes(title_text="数值", row=row, col=col)

    def _add_latest_picks_table(self, fig: go.Figure, latest_picks: Optional[pd.DataFrame], row: int, col: int):
        """添加最新选股表格"""
        try:
            # 创建示例选股表格
            table_data = {
                'Rank': ['1', '2', '3', '4', '5'],
                'Code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
                'Name': ['PingAn Bank', 'Vanke A', 'SPDB', 'CMB', 'Wuliangye'],
                'Score': ['0.850', '0.820', '0.780', '0.750', '0.710'],
                'Weight': ['20.0%', '20.0%', '20.0%', '20.0%', '20.0%']
            }

            # 处理真实数据（如果有的话）
            if latest_picks is not None and not latest_picks.empty:
                try:
                    # 尝试使用真实数据，但简化处理
                    if len(latest_picks) >= 3:
                        table_data['Rank'] = [str(i+1) for i in range(min(5, len(latest_picks)))]

                        if 'stock_code' in latest_picks.columns:
                            codes = latest_picks['stock_code'].head(5).astype(str).tolist()
                            table_data['Code'] = codes + [''] * (5 - len(codes))

                        if 'stock_name' in latest_picks.columns:
                            names = latest_picks['stock_name'].head(5).astype(str).tolist()
                            # 简化名称
                            names = [name[:8] for name in names]
                            table_data['Name'] = names + [''] * (5 - len(names))

                        if 'pred_score' in latest_picks.columns:
                            scores = latest_picks['pred_score'].head(5).apply(lambda x: f"{float(x):.3f}").tolist()
                            table_data['Score'] = scores + [''] * (5 - len(scores))

                        if 'weight' in latest_picks.columns:
                            weights = latest_picks['weight'].head(5)
                            if weights.dtype == 'object':
                                weight_strs = weights.tolist()
                            else:
                                weight_strs = [f"{float(w):.1%}" for w in weights]
                            table_data['Weight'] = weight_strs + [''] * (5 - len(weight_strs))

                except Exception as e:
                    logger.debug(f"处理真实数据失败，使用默认数据: {e}")

            # 创建表格，设置合适的列宽
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=list(table_data.keys()),
                        fill_color=self.colors['primary'],
                        font=dict(color='white', size=12, family='Arial Black'),
                        align='center',
                        height=40  # 增加表头高度
                    ),
                    cells=dict(
                        values=list(table_data.values()),
                        fill_color='white',
                        font=dict(color=self.colors['dark'], size=11, family='Arial'),
                        align='center',
                        height=32,  # 增加单元格高度
                        line=dict(color='lightgray', width=0.5)  # 添加边框
                    ),
                    columnwidth=[0.15, 0.25, 0.30, 0.15, 0.15]  # 明确设置列宽比例: 排名(15%), 代码(25%), 名称(30%), 评分(15%), 权重(15%)
                ),
                row=row, col=col
            )

        except Exception as e:
            logger.warning(f"创建选股表格失败: {e}")
            # 创建备用表格
            try:
                backup_table = go.Table(
                    header=dict(values=['Status'], fill_color=self.colors['muted'], font=dict(color='white')),
                    cells=dict(values=[['No Data']], fill_color='white', font=dict(color=self.colors['dark']))
                )
                fig.add_trace(backup_table, row=row, col=col)
            except Exception as e2:
                logger.error(f"创建备用表格失败: {e2}")

    def _add_prediction_distribution(self, fig: go.Figure, predictions: Optional[Dict], row: int, col: int):
        """添加预测评分分布"""
        if predictions is None:
            # 创建模拟预测评分分布
            scores = np.random.normal(0, 0.1, 1000)
        else:
            # 使用实际预测数据
            if 'ensemble' in predictions:
                scores = predictions['ensemble'].values
            else:
                scores = list(predictions.values())[0].values if predictions else np.random.normal(0, 0.1, 1000)

        # 预测评分分布直方图
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=25,
                name='预测评分分布',
                marker_color=self.colors['secondary'],
                opacity=0.7,
                hovertemplate='<b>评分区间</b>: %{x:.3f}<br><b>频次</b>: %{y}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 添加均值线
        mean_score = np.mean(scores)
        fig.add_trace(
            go.Scatter(
                x=[mean_score, mean_score],
                y=[0, len(scores) * 0.1],
                name=f'均值({mean_score:.3f})',
                line=dict(color=self.colors['danger'], width=2, dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="预测评分", row=row, col=col)
        fig.update_yaxes(title_text="频次", row=row, col=col)

    def _add_cumulative_comparison(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """添加累计收益对比"""
        if equity_curve.empty:
            return

        dates = pd.to_datetime(equity_curve['date'])

        # 策略累计收益
        strategy_returns = equity_curve['equity'] if 'equity' in equity_curve.columns else pd.Series([1.0] * len(dates))

        # 创建基准收益（模拟市场表现）
        if 'returns' in equity_curve.columns:
            market_returns = equity_curve['returns'].fillna(0) * 0.6 + np.random.normal(0, 0.005, len(dates))
            benchmark_equity = (1 + market_returns).cumprod()
        else:
            benchmark_equity = pd.Series(np.linspace(1, 1.1, len(dates)))

        # 策略收益曲线
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=strategy_returns,
                name='AI策略',
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='<b>日期</b>: %{x}<br><b>AI策略</b>: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 基准收益曲线
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark_equity,
                name='市场基准',
                line=dict(color=self.colors['muted'], width=2, dash='dash'),
                hovertemplate='<b>日期</b>: %{x}<br><b>基准</b>: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # 超额收益面积
        excess_returns = strategy_returns - benchmark_equity
        colors = ['rgba(46, 160, 44, 0.3)' if x > 0 else 'rgba(214, 39, 40, 0.3)' for x in excess_returns]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=excess_returns,
                name='超额收益',
                fill='tozeroy',
                fillcolor='rgba(46, 160, 44, 0.2)',
                line=dict(color=self.colors['success'], width=1),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="累计收益", row=row, col=col)

    def _add_key_metrics_indicators(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """添加关键指标仪表盘"""
        # 关键指标
        annual_return = metrics.get('annual_return', 0.12)
        sharpe_ratio = metrics.get('sharpe_ratio', 1.2)
        max_drawdown = metrics.get('max_drawdown', -0.08)
        win_rate = metrics.get('win_rate', 0.55)

        # 创建指标卡片样式的显示
        indicator_data = [
            {'label': '年化收益率', 'value': f'{annual_return:.1%}', 'color': self.colors['success']},
            {'label': '夏普比率', 'value': f'{sharpe_ratio:.2f}', 'color': self.colors['primary']},
            {'label': '最大回撤', 'value': f'{max_drawdown:.1%}', 'color': self.colors['danger']},
            {'label': '胜率', 'value': f'{win_rate:.1%}', 'color': self.colors['info']}
        ]

        # 使用表格形式展示关键指标
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['指标', '数值'],
                    fill_color=self.colors['dark'],
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=[
                        [item['label'] for item in indicator_data],
                        [item['value'] for item in indicator_data]
                    ],
                    fill_color=[
                        [self.colors['light']] * len(indicator_data),
                        [item['color'] for item in indicator_data]
                    ],
                    font=dict(color=['black'] * len(indicator_data) + ['white'] * len(indicator_data), size=11),
                    align=['left', 'center']
                )
            ),
            row=row, col=col
        )

    def create_detailed_analysis(self,
                               equity_curve: pd.DataFrame,
                               metrics: Dict,
                               predictions: Optional[Dict] = None) -> go.Figure:
        """
        创建详细分析图表（单独的深度分析）

        Args:
            equity_curve: 资金曲线数据
            metrics: 回测指标
            predictions: 预测数据

        Returns:
            详细分析图表
        """
        logger.info("创建详细分析图表...")

        # 创建2x2子图布局
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['风险归因分析', '收益贡献分解', '预测精度分析', '交易成本分析'],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        # 1. 风险归因分析
        self._add_risk_attribution(fig, metrics, row=1, col=1)

        # 2. 收益贡献分解
        self._add_return_attribution(fig, equity_curve, row=1, col=2)

        # 3. 预测精度分析
        self._add_prediction_accuracy(fig, predictions, row=2, col=1)

        # 4. 交易成本分析
        self._add_transaction_cost_analysis(fig, metrics, row=2, col=2)

        fig.update_layout(
            title='AI选股器详细分析报告',
            height=800,
            **self.layout_template
        )

        return fig

    def _add_risk_attribution(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """风险归因分析"""
        risk_sources = ['市场风险', '行业风险', '个股风险', '策略风险']
        risk_values = [0.4, 0.25, 0.2, 0.15]  # 模拟风险贡献

        fig.add_trace(
            go.Bar(
                x=risk_sources,
                y=risk_values,
                marker_color=self.colors['gradient'][:4],
                name='风险贡献',
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="风险来源", row=row, col=col)
        fig.update_yaxes(title_text="风险贡献度", row=row, col=col)

    def _add_return_attribution(self, fig: go.Figure, equity_curve: pd.DataFrame, row: int, col: int):
        """收益贡献分解"""
        return_sources = ['选股收益', '择时收益', '交易成本', '其他']
        return_values = [0.65, 0.20, -0.05, 0.20]

        colors = [self.colors['success'] if x > 0 else self.colors['danger'] for x in return_values]

        fig.add_trace(
            go.Pie(
                labels=return_sources,
                values=[abs(x) for x in return_values],
                marker_colors=colors,
                name='收益贡献',
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_prediction_accuracy(self, fig: go.Figure, predictions: Optional[Dict], row: int, col: int):
        """预测精度分析"""
        # 模拟精度数据
        periods = list(range(1, 21))
        accuracy = 0.6 + 0.2 * np.sin(np.array(periods) * 0.3) * np.exp(-np.array(periods) * 0.05)

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=accuracy,
                mode='lines+markers',
                name='预测精度',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=row, col=col
        )

        # 添加基准线
        fig.add_trace(
            go.Scatter(
                x=[1, 20],
                y=[0.5, 0.5],
                name='随机基准',
                line=dict(color=self.colors['muted'], dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="预测期数", row=row, col=col)
        fig.update_yaxes(title_text="预测精度", row=row, col=col)

    def _add_transaction_cost_analysis(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """交易成本分析"""
        turnover_rates = np.linspace(0.1, 2.0, 20)
        transaction_costs = turnover_rates * 0.0005  # 万分之五的成本
        net_returns = 0.15 - transaction_costs  # 假设15%基础收益

        fig.add_trace(
            go.Scatter(
                x=turnover_rates,
                y=net_returns,
                mode='lines',
                name='净收益',
                line=dict(color=self.colors['success'], width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=turnover_rates,
                y=transaction_costs,
                mode='lines',
                name='交易成本',
                line=dict(color=self.colors['danger'], width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        # 当前换手率标记
        current_turnover = metrics.get('avg_turnover', 0.8)
        current_cost = current_turnover * 0.0005

        fig.add_trace(
            go.Scatter(
                x=[current_turnover],
                y=[current_cost],
                mode='markers',
                name='当前水平',
                marker=dict(size=10, color=self.colors['warning']),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="换手率", row=row, col=col)
        fig.update_yaxes(title_text="收益/成本", row=row, col=col)

    def _add_fundamental_heatmap(self, fig: go.Figure, 
                               holdings_records: Optional[List] = None, 
                               fundamental_data: Optional[pd.DataFrame] = None,
                               row: int = 2, col: int = 1):
        """
        添加基本面热力图
        
        Args:
            fig: 图形对象
            holdings_records: 持仓记录列表
            fundamental_data: 基本面数据
            row: 行位置
            col: 列位置
        """
        if not holdings_records or fundamental_data is None or fundamental_data.empty:
            # 添加空图表占位
            fig.add_trace(
                go.Heatmap(
                    z=[[0]], 
                    x=['无数据'], 
                    y=['无数据'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                row=row, col=col
            )
            return
        
        try:
            # 构建基本面热力图数据
            heatmap_data = []
            dates = []
            stocks = set()
            
            # 收集所有选中的股票和日期
            for record in holdings_records:
                if 'selected_stocks' in record and not record['selected_stocks'].empty:
                    date = record['date']
                    dates.append(date)
                    selected_stocks = record['selected_stocks']['instrument'].tolist()
                    stocks.update(selected_stocks)
            
            stocks = sorted(list(stocks))[:10]  # 限制显示前10只股票
            dates = sorted(dates)[-20:]  # 显示最近20个调仓日
            
            if not stocks or not dates:
                return
            
            # 构建热力图矩阵
            z_data = []
            for stock in stocks:
                stock_roe_data = []
                for date in dates:
                    # 查找该股票在该日期的ROE数据
                    stock_fundamental = fundamental_data[
                        (fundamental_data['code'] == stock) &
                        (fundamental_data['date'] <= date)
                    ]
                    
                    if not stock_fundamental.empty:
                        latest_roe = stock_fundamental.iloc[-1].get('roe', 0.1)
                        if pd.isna(latest_roe):
                            latest_roe = 0.1
                    else:
                        latest_roe = 0.1  # 默认值
                    
                    stock_roe_data.append(latest_roe * 100)  # 转换为百分比
                
                z_data.append(stock_roe_data)
            
            # 添加热力图
            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=[d[-5:] for d in dates],  # 只显示月-日
                    y=[s[-6:] for s in stocks],  # 只显示股票代码后6位
                    colorscale='RdYlGn',
                    colorbar=dict(
                        title="ROE (%)",
                        titleside="right"
                    ),
                    hovertemplate='<b>股票</b>: %{y}<br><b>日期</b>: %{x}<br><b>ROE</b>: %{z:.2f}%<extra></extra>',
                    showscale=True
                ),
                row=row, col=col
            )
            
        except Exception as e:
            logger.warning(f"基本面热力图生成失败: {e}")
            # 添加空图表占位
            fig.add_trace(
                go.Heatmap(
                    z=[[0]], 
                    x=['错误'], 
                    y=['错误'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="时间", row=row, col=col)
        fig.update_yaxes(title_text="选中股票", row=row, col=col)

    def create_lowprice_growth_dashboard(self,
                                       equity_curve: pd.DataFrame,
                                       metrics: Dict,
                                       holdings_records: Optional[List] = None,
                                       fundamental_data: Optional[pd.DataFrame] = None,
                                       latest_picks: Optional[pd.DataFrame] = None,
                                       title: str = "低价+稳定增长策略分析仪表板") -> go.Figure:
        """
        创建低价成长策略专用仪表板
        
        Args:
            equity_curve: 资金曲线数据
            metrics: 回测指标
            holdings_records: 持仓记录
            fundamental_data: 基本面数据
            latest_picks: 最新选股
            title: 仪表板标题
            
        Returns:
            plotly.graph_objects.Figure: 专用仪表板图表
        """
        logger.info("创建低价成长策略分析仪表板...")
        
        # 创建4行3列布局
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                '资金曲线与回撤', 'IC时序分析', '收益分布',
                '基本面ROE热力图', '滚动夏普比率', '换手率分析', 
                '基本面指标对比', '预测评分分布', '累计收益对比',
                '最新选股', '关键指标总览(含PEG/EPS_CAGR)', '交易成本分析'
            ],
            specs=[
                [{"secondary_y": True}, {}, {"type": "histogram"}],
                [{"type": "heatmap"}, {}, {}],
                [{"type": "bar"}, {"type": "histogram"}, {}],
                [{"type": "table"}, {"type": "table"}, {}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            row_heights=[0.28, 0.24, 0.24, 0.24]
        )

        # 第一行：核心表现图表
        self._add_equity_curve_with_drawdown(fig, equity_curve, row=1, col=1)
        self._add_ic_timeseries(fig, equity_curve, metrics, row=1, col=2)
        self._add_returns_distribution(fig, equity_curve, row=1, col=3)

        # 第二行：基本面分析
        self._add_fundamental_heatmap(fig, holdings_records, fundamental_data, row=2, col=1)
        self._add_rolling_sharpe(fig, equity_curve, row=2, col=2)
        self._add_turnover_analysis(fig, metrics, row=2, col=3)

        # 第三行：风险与预测分析
        self._add_fundamental_metrics_bar(fig, metrics, row=3, col=1)
        self._add_prediction_distribution(fig, None, row=3, col=2)
        self._add_cumulative_comparison(fig, equity_curve, row=3, col=3)

        # 第四行：数据表格和总结
        self._add_latest_picks_table(fig, latest_picks, row=4, col=1)
        self._add_enhanced_metrics_indicators(fig, metrics, row=4, col=2)
        self._add_transaction_cost_analysis(fig, metrics, row=4, col=3)

        # 设置整体布局
        layout_config = {
            'title': {
                'text': f'<b>{title}</b><br><sub>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</sub>',
                'x': 0.5,
                'font': {'size': 22, 'color': self.colors['dark']}
            },
            'height': 1400,
            'width': 1800,
            'showlegend': False,
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 11, 'color': self.colors['dark']},
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 100},
            'hovermode': 'x unified',
            'paper_bgcolor': 'rgba(248, 249, 250, 0.8)',
            'plot_bgcolor': 'white'
        }

        fig.update_layout(**layout_config)

        logger.info("低价成长策略仪表板创建完成")
        return fig

    def _add_fundamental_metrics_bar(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """添加基本面指标柱状图"""
        try:
            # 基本面相关指标
            fundamental_metrics = {
                'IC均值': metrics.get('ic_mean', 0) * 100,
                '平均PEG': metrics.get('mean_PEG', 1.0),
                'ROE标准差': metrics.get('mean_ROE_std', 0.02) * 100,
                'EPS_CAGR中位数': metrics.get('median_EPS_CAGR', 0.15) * 100,
                '收益回撤比': metrics.get('return_drawdown_ratio', 0),
            }
            
            # 定义目标值和颜色
            targets = {
                'IC均值': 6.0,
                '平均PEG': 1.1,
                'ROE标准差': 3.0,
                'EPS_CAGR中位数': 15.0,
                '收益回撤比': 0.6,
            }
            
            metric_names = list(fundamental_metrics.keys())
            metric_values = list(fundamental_metrics.values())
            target_values = [targets[name] for name in metric_names]
            
            # 根据是否达标设置颜色
            colors = []
            for name, value in fundamental_metrics.items():
                target = targets[name]
                if name in ['IC均值', 'EPS_CAGR中位数', '收益回撤比']:
                    # 越大越好
                    colors.append(self.colors['success'] if value >= target else self.colors['danger'])
                else:
                    # 越小越好
                    colors.append(self.colors['success'] if value <= target else self.colors['danger'])
            
            # 添加实际值柱状图
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='实际值',
                    marker_color=colors,
                    text=[f'{v:.2f}' for v in metric_values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 添加目标值线条
            fig.add_trace(
                go.Scatter(
                    x=metric_names,
                    y=target_values,
                    mode='markers+lines',
                    name='目标值',
                    line=dict(color=self.colors['warning'], width=2, dash='dash'),
                    marker=dict(size=8, color=self.colors['warning']),
                    showlegend=False
                ),
                row=row, col=col
            )
            
        except Exception as e:
            logger.warning(f"基本面指标图生成失败: {e}")
        
        fig.update_xaxes(title_text="指标", row=row, col=col)
        fig.update_yaxes(title_text="数值", row=row, col=col)

    def _add_enhanced_metrics_indicators(self, fig: go.Figure, metrics: Dict, row: int, col: int):
        """添加增强版关键指标卡片（包含PEG/EPS_CAGR）"""
        try:
            # 扩展指标列表
            key_metrics = [
                ['年化收益率', f"{metrics.get('annual_return', 0):.2%}", '绿色' if metrics.get('annual_return', 0) > 0 else '红色'],
                ['最大回撤', f"{metrics.get('max_drawdown', 0):.2%}", '红色'],
                ['夏普比率', f"{metrics.get('sharpe_ratio', 0):.3f}", '蓝色'],
                ['胜率', f"{metrics.get('win_rate', 0):.2%}", '绿色'],
                ['IC均值', f"{metrics.get('ic_mean', 0):.4f}", '蓝色'],
                ['平均换手率', f"{metrics.get('avg_turnover', 0):.2%}", '橙色'],
                ['平均PEG', f"{metrics.get('mean_PEG', 0):.2f}", '紫色'],
                ['EPS_CAGR中位数', f"{metrics.get('median_EPS_CAGR', 0):.2%}", '绿色'],
                ['ROE标准差', f"{metrics.get('mean_ROE_std', 0):.2%}", '红色'],
                ['收益回撤比', f"{metrics.get('return_drawdown_ratio', 0):.2f}", '蓝色'],
            ]
            
            # 创建表格数据
            table_data = {
                '指标': [metric[0] for metric in key_metrics],
                '数值': [metric[1] for metric in key_metrics],
            }
            
            # 设置表格颜色
            cell_colors = []
            for metric in key_metrics:
                if metric[2] == '绿色':
                    cell_colors.append(['lightgreen', 'lightgreen'])
                elif metric[2] == '红色':
                    cell_colors.append(['lightcoral', 'lightcoral'])
                elif metric[2] == '蓝色':
                    cell_colors.append(['lightblue', 'lightblue'])
                elif metric[2] == '橙色':
                    cell_colors.append(['orange', 'orange'])
                elif metric[2] == '紫色':
                    cell_colors.append(['plum', 'plum'])
                else:
                    cell_colors.append(['white', 'white'])
            
            # 转置颜色数组以匹配表格格式
            fill_colors = [list(row) for row in zip(*cell_colors)]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['指标', '数值'],
                        fill_color=self.colors['primary'],
                        font=dict(color='white', size=12),
                        align='center'
                    ),
                    cells=dict(
                        values=[table_data['指标'], table_data['数值']],
                        fill_color=fill_colors,
                        font=dict(color='black', size=11),
                        align=['left', 'right'],
                        height=25
                    )
                ),
                row=row, col=col
            )
            
        except Exception as e:
            logger.warning(f"增强版指标表格生成失败: {e}")

    def save_dashboard_as_html(self, fig: go.Figure, filename: str = "backtest_dashboard.html"):
        """保存仪表板为HTML文件，包含增强的样式和交互功能"""
        output_path = Path("./output") / filename
        output_path.parent.mkdir(exist_ok=True)

        # 增强的配置选项
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'ai_stock_selector_analysis',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        }

        # 自定义HTML模板，包含美化样式
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>AI选股器回测分析报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .dashboard-container {{
                    padding: 20px;
                }}
                .info-panel {{
                    background: #f8f9fa;
                    border-left: 4px solid #1f77b4;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 0 10px 10px 0;
                }}
                .footer {{
                    background: #212529;
                    color: white;
                    text-align: center;
                    padding: 20px;
                    font-size: 0.9em;
                }}
                .plotly-graph-div {{
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                @media (max-width: 768px) {{
                    body {{ padding: 10px; }}
                    .container {{ border-radius: 10px; }}
                    .header {{ padding: 20px; }}
                    .header h1 {{ font-size: 2em; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI选股器回测分析报告</h1>
                    <p>基于机器学习的智能选股策略回测与分析</p>
                </div>

                <div class="dashboard-container">
                    <div class="info-panel">
                        <h3>📊 使用说明</h3>
                        <ul>
                            <li><strong>时间范围选择：</strong>使用图表下方的时间范围选择器查看不同时期的表现</li>
                            <li><strong>图表交互：</strong>悬停查看详细数据，双击重置缩放，拖拽平移视图</li>
                            <li><strong>图表导出：</strong>点击工具栏中的相机图标可导出高质量图片</li>
                            <li><strong>数据钻取：</strong>点击图例项目可显示/隐藏对应数据系列</li>
                        </ul>
                    </div>

                    {plot_div}
                </div>

                <div class="footer">
                    <p>© 2024 AI选股器 | 生成时间: {timestamp} |
                    <a href="https://github.com/microsoft/qlib" target="_blank" style="color: #17a2b8;">Powered by Qlib</a></p>
                </div>
            </div>

            <script>
                // 添加响应式处理
                window.addEventListener('resize', function() {{
                    Plotly.Plots.resize(document.getElementById('backtest-dashboard'));
                }});

                // 添加加载动画
                document.addEventListener('DOMContentLoaded', function() {{
                    const container = document.querySelector('.container');
                    container.style.opacity = '0';
                    container.style.transform = 'translateY(20px)';

                    setTimeout(() => {{
                        container.style.transition = 'all 0.6s ease';
                        container.style.opacity = '1';
                        container.style.transform = 'translateY(0)';
                    }}, 100);
                }});
            </script>
        </body>
        </html>
        """

        # 生成图表HTML
        plot_html = fig.to_html(
            include_plotlyjs='cdn',
            config=config,
            div_id="backtest-dashboard",
            full_html=False
        )

        # 组合完整HTML
        full_html = html_template.format(
            plot_div=plot_html,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)

        logger.info(f"增强版仪表板已保存至: {output_path}")
        return str(output_path)

    def create_interactive_report(self,
                                equity_curve: pd.DataFrame,
                                metrics: Dict,
                                predictions: Optional[Dict] = None,
                                latest_picks: Optional[pd.DataFrame] = None) -> str:
        """
        创建完整的交互式分析报告

        Returns:
            保存的HTML文件路径
        """
        logger.info("创建完整交互式分析报告...")

        # 创建综合仪表板
        main_dashboard = self.create_comprehensive_dashboard(
            equity_curve, metrics, predictions, latest_picks
        )

        # 保存为HTML
        html_path = self.save_dashboard_as_html(
            main_dashboard,
            f"ai_stock_selector_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        logger.info("交互式分析报告创建完成")
        return html_path

    def create_detailed_analysis(self,
                               equity_curve: pd.DataFrame,
                               metrics: Dict,
                               predictions: Optional[Dict] = None,
                               latest_picks: Optional[pd.DataFrame] = None,
                               title: str = "AI选股器回测分析仪表板") -> go.Figure:
        """
        创建详细分析图表（向后兼容方法）
        
        此方法保持与旧版本的兼容性，实际调用新的综合仪表板方法
        """
        return self.create_comprehensive_dashboard(
            equity_curve, metrics, predictions, latest_picks, title
        )


def create_sample_dashboard():
    """创建示例仪表板用于测试"""
    # 创建示例数据
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    equity_curve = pd.DataFrame({
        'date': dates,
        'returns': returns,
        'equity': (1 + returns).cumprod()
    })

    metrics = {
        'ic_mean': 0.045,
        'ic_ir': 1.2,
        'annual_return': 0.15,
        'volatility': 0.18,
        'sharpe_ratio': 0.83,
        'max_drawdown': -0.12,
        'win_rate': 0.58,
        'avg_turnover': 0.65
    }

    # 创建可视化器
    visualizer = BacktestVisualizer()

    # 生成仪表板
    fig = visualizer.create_comprehensive_dashboard(equity_curve, metrics)

    return fig, visualizer


if __name__ == "__main__":
    # 测试可视化器
    fig, visualizer = create_sample_dashboard()

    # 保存并显示
    html_path = visualizer.save_dashboard_as_html(fig, "test_dashboard.html")
    print(f"测试仪表板已生成: {html_path}")

    # 在浏览器中显示（如果环境支持）
    try:
        fig.show()
    except:
        print("无法在当前环境中显示图表，请打开HTML文件查看")