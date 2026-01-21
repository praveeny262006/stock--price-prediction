import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff

def render_stock_visualizations(results):
    """Render stock prediction visualizations with focus on trading decisions"""
    try:
        # Validate input structure
        required_keys = ['ticker', 'historical_data', 'prediction', 'evaluation', 'dates']
        if not all(k in results for k in required_keys):
            st.error(f"Missing required data in results. Expected keys: {required_keys}")
            return

        ticker = results['ticker']
        hist_data = results['historical_data']
        pred = results['prediction']
        eval_data = results['evaluation']
        dates = results['dates']

        # Color scheme
        color_scheme = {
            'actual': '#00FF00',
            'predicted': '#FF0000',
            'up': '#00FF00',
            'down': '#FF0000',
            'accuracy_high': '#00FF00',
            'accuracy_med': '#FFA500',
            'accuracy_low': '#FF0000'
        }

        # 1. Prediction Summary Cards
        st.subheader("Trading Recommendation")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Last Close Price", f"${pred.get('last_close', 0):.2f}")
        with cols[1]:
            price_diff = pred.get('price', 0) - pred.get('last_close', 0)
            st.metric("Predicted Price", 
                     f"${pred.get('price', 0):.2f}",
                     f"{price_diff:.2f} ({price_diff/pred.get('last_close', 1)*100:.2f}%)")
        with cols[2]:
            direction = pred.get('direction', 'UNKNOWN')
            rec = "BUY" if direction == "UP" else "SELL"
            st.metric("Recommendation", rec, delta_color="off")

        # 2. Actual vs Predicted Comparison
        st.subheader("Model Performance: Actual vs Predicted")
        try:
            if isinstance(eval_data['regression']['actual'], np.ndarray):
                eval_data['regression']['actual'] = pd.Series(
                    eval_data['regression']['actual'],
                    index=pd.to_datetime(dates['test_dates'])
                )

            if isinstance(eval_data['regression']['predicted'], np.ndarray):
                eval_data['regression']['predicted'] = pd.Series(
                    eval_data['regression']['predicted'],
                    index=pd.to_datetime(dates['test_dates'])
                )

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=eval_data['regression']['actual'].index,
                y=eval_data['regression']['actual'],
                name='Actual Price',
                line=dict(color=color_scheme['actual'], width=2),
                mode='lines+markers'
            ))
            fig2.add_trace(go.Scatter(
                x=eval_data['regression']['predicted'].index,
                y=eval_data['regression']['predicted'],
                name='Predicted Price',
                line=dict(color=color_scheme['predicted'], width=2, dash='dash'),
                mode='lines+markers'
            ))
            fig2.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Actual vs Predicted: {str(e)}")

        # 3. Volatility & Price Change Trend
        st.subheader("Volatility & Price Change Trend")
        try:
            actual_prices = eval_data['regression']['actual']
            returns = actual_prices.pct_change().dropna() * 100
            volatility = returns.rolling(window=5).std()

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=returns.index, y=returns,
                name='Daily Return (%)',
                line=dict(color='orange', width=2),
                mode='lines+markers',
                yaxis='y1'
            ))
            fig_vol.add_trace(go.Scatter(
                x=volatility.index, y=volatility,
                name='Rolling Volatility (5D)',
                line=dict(color='purple', width=2, dash='dot'),
                mode='lines',
                yaxis='y2'
            ))
            fig_vol.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Date',
                yaxis=dict(title='Daily Return (%)', side='left', showgrid=False),
                yaxis2=dict(title='Volatility', overlaying='y', side='right', showgrid=False),
                hovermode='x unified',
                legend=dict(x=0, y=1.15, orientation='h')
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Volatility Plot: {str(e)}")

        # 4. Moving Average Crossover
        st.subheader("Moving Average Crossover")
        try:
            if isinstance(hist_data, pd.DataFrame):
                # Using dynamic column selection to handle MultiIndex or Flat Index
                ma_short = hist_data.get(('MA_5', '')) if ('MA_5', '') in hist_data.columns else hist_data.get('MA_5')
                ma_long = hist_data.get(('MA_20', '')) if ('MA_20', '') in hist_data.columns else hist_data.get('MA_20')
                prices = hist_data.iloc[:, 0] # Default to first column (usually Close)

                if ma_short is not None and ma_long is not None:
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='#1f77b4', width=1)))
                    fig_ma.add_trace(go.Scatter(x=ma_short.index, y=ma_short, name='5-Day MA', line=dict(color=color_scheme['up'], width=2)))
                    fig_ma.add_trace(go.Scatter(x=ma_long.index, y=ma_long, name='20-Day MA', line=dict(color=color_scheme['down'], width=2)))
                    
                    crossover_up = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
                    crossover_down = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
                    
                    fig_ma.add_trace(go.Scatter(x=prices.index[crossover_up], y=prices[crossover_up], name='Buy Signal', mode='markers', marker=dict(color=color_scheme['up'], size=10, symbol='triangle-up')))
                    fig_ma.add_trace(go.Scatter(x=prices.index[crossover_down], y=prices[crossover_down], name='Sell Signal', mode='markers', marker=dict(color=color_scheme['down'], size=10, symbol='triangle-down')))
                    
                    fig_ma.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), hovermode='x unified')
                    st.plotly_chart(fig_ma, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Moving Average Plot: {str(e)}")

        # 5. RSI Indicator
        st.subheader("RSI Indicator")
        try:
            rsi = hist_data.get(('RSI', '')) if ('RSI', '') in hist_data.columns else hist_data.get('RSI')
            if rsi is not None:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='#FFA500', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color=color_scheme['down'], annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color=color_scheme['up'], annotation_text="Oversold")
                fig_rsi.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)
        except Exception as e:
            st.error(f"Error in RSI Plot: {str(e)}")

        # 6. Model Metrics (Regression & Classification)
        st.subheader("Model Performance Metrics")
        try:
            actual_prices = eval_data['regression']['actual']
            predicted_prices = eval_data['regression']['predicted']
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            
            y_true = eval_data['classification']['actual']
            y_pred = eval_data['classification']['predicted']
            
            st.markdown("""
                <style>
                .metric-card { border: 1px solid rgba(255,255,255,0.1); border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; background-color: rgba(0,0,0,0.2); }
                .metric-title { font-size: 0.9rem; color: rgba(255,255,255,0.6); }
                .metric-value { font-size: 1.5rem; font-weight: bold; color: #FFFFFF; }
                </style>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">MAPE (Regression Error)</div><div class="metric-value">{mape:.2f}%</div></div>', unsafe_allow_html=True)
            with c2:
                acc = accuracy_score(y_true, y_pred) * 100
                st.markdown(f'<div class="metric-card"><div class="metric-title">Directional Accuracy</div><div class="metric-value">{acc:.1f}%</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")

        # 7. Confusion Matrix
        try:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = ff.create_annotated_heatmap(z=cm.tolist(), x=["Down", "Up"], y=["Down", "Up"], colorscale='Blues')
            fig_cm.update_layout(font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm, use_container_width=True)
        except: pass

        # --- FOOTER SECTION ---
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 1.1rem; font-weight: 500;">
                    Developed By <span style="color: #4FC3F7; font-weight: bold;">Praveen Y</span>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
