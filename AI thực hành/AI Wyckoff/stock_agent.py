# =============================================================================
# OPTIMIZED WYCKOFF METHOD ANALYSIS - CREWAI SYSTEM
# Combines: File-based efficiency + Comprehensive Wyckoff theory + Intraday Analysis
# =============================================================================
import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from enum import Enum
from vnstock import Quote
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

# =============================================================================
# CONFIGURATION
# =============================================================================
WORK_DIR = os.path.join(os.getcwd(), "wyckoff_analysis")
os.makedirs(WORK_DIR, exist_ok=True)

def get_path(filename: str) -> str:
    return os.path.join(WORK_DIR, filename)

# =============================================================================
# STATE & ERROR HANDLING
# =============================================================================
class WyckoffState:
    """Compact state object for passing between agents"""
    def __init__(self):
        # Current market data
        self.current_price: float = 0.0
        self.current_date: str = ""
        self.data_points: int = 0
        
        # Wyckoff analysis
        self.schematic: str = "UNKNOWN"  # Accumulation/Distribution
        self.phase: str = "UNKNOWN"      # Phase A/B/C/D/E
        self.support: float = 0.0
        self.resistance: float = 0.0
        
        # Wyckoff Laws scores
        self.supply_demand_score: float = 0.0  # Law 1
        self.cause_effect_range: float = 0.0   # Law 2
        self.effort_result_score: float = 0.0  # Law 3
        
        # Events & Trading (store as simple strings, not complex objects)
        self.event_count: int = 0
        self.entry_low: float = 0.0
        self.entry_high: float = 0.0
        self.stop_loss: float = 0.0
        self.target1: float = 0.0
        self.target2: float = 0.0
        
        # NEW: Intraday metrics
        self.intraday_records: int = 0
        self.intraday_absorption_score: float = 0.0
        self.intraday_dominant_force: str = "NEUTRAL"
        self.intraday_institutional_signals: int = 0
        self.climactic_volume_events: int = 0
        self.no_demand_bars: int = 0
        self.no_supply_bars: int = 0
        self.stopping_volume_bars: int = 0
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str):
        obj = cls()
        try:
            # Handle both string and dict inputs
            if isinstance(json_str, dict):
                obj.__dict__.update(json_str)
            else:
                obj.__dict__.update(json.loads(json_str))
        except Exception as e:
            print(f"⚠️ State parse warning: {e}")
        return obj

class ErrorSeverity(Enum):
    WARNING = "WARN"
    CRITICAL = "CRIT"

class WyckoffError(Exception):
    def __init__(self, tool: str, severity: ErrorSeverity, msg: str):
        super().__init__(f"{severity.value}|{tool}|{msg}")

# =============================================================================
# TOOLS - ORIGINAL (UNCHANGED)
# =============================================================================
class WyckoffDataTool(BaseTool):
    name: str = "Data Fetcher"
    description: str = "Fetches HPG + VNINDEX data and saves to file"

    def _run(self) -> str:
        try:
            print("🔍 Fetching VNINDEX...")
            vnindex = Quote(symbol='VNINDEX', source='VCI').history(start='2025-01-01', end='2026-01-14')
            vnindex.rename(columns={'time':'Date','close':'VNIndex_Close'}, inplace=True)
            vnindex['Date'] = pd.to_datetime(vnindex['Date'])
            vnindex = vnindex[['Date','VNIndex_Close']]

            print("🔍 Fetching HPG...")
            data = Quote(symbol='HPG', source='VCI').history(start='2025-01-01', end='2026-01-14')
            
            if data.empty:
                raise WyckoffError(self.name, ErrorSeverity.CRITICAL, "No data")

            df = pd.DataFrame(data)
            df.rename(columns={
                'time': 'Date', 'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.merge(vnindex, on='Date', how='left')
            
            state = WyckoffState()
            state.current_price = float(df['Close'].iloc[-1])
            state.current_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
            state.data_points = len(df)
            
            # Save file
            file_path = get_path("data.csv")
            df.to_csv(file_path, index=False)
            
            if not os.path.exists(file_path):
                raise WyckoffError(self.name, ErrorSeverity.CRITICAL, "Save failed")
            
            print(f"✅ Saved {len(df)} rows to: {file_path}")
            return f"STATE:{state.to_json()}\nFILE:{file_path}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))


class WyckoffLawAnalyzer(BaseTool):
    name: str = "Law Analyzer"
    description: str = "Analyzes 3 Wyckoff Laws: Supply/Demand, Cause/Effect, Effort/Result"
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            df = pd.read_csv(get_path("data.csv"))
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Calculate indicators
            df['Price_Change'] = df['Close'].diff()
            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            df['High_Low_Range'] = df['High'] - df['Low']
            
            # LAW 1: Supply & Demand (comparing buying vs selling pressure)
            demand = (df['Close'] - df['Low']).iloc[-20:].mean()  # Closes near highs = demand
            supply = (df['High'] - df['Close']).iloc[-20:].mean()  # Closes near lows = supply
            state.supply_demand_score = (demand - supply) / (demand + supply) * 100
            
            # LAW 2: Cause & Effect (trading range predicts move)
            recent_high = df['High'].iloc[-60:].max()
            recent_low = df['Low'].iloc[-60:].min()
            state.cause_effect_range = recent_high - recent_low
            
            # Ensure we always have valid support/resistance
            if state.cause_effect_range > 0:
                state.support = float(recent_low)
                state.resistance = float(recent_high)
            else:
                # Fallback: use current price with buffer
                current = state.current_price
                state.support = current * 0.95
                state.resistance = current * 1.05
                state.cause_effect_range = state.resistance - state.support
            
            # LAW 3: Effort vs Result (volume should match price movement)
            effort_result = []
            for i in range(-20, 0):
                vol_high = df['Volume_Ratio'].iloc[i] > 1.5
                price_move = abs(df['Price_Change'].iloc[i])
                range_size = df['High_Low_Range'].iloc[i]
                
                if vol_high and price_move < range_size * 0.3:
                    effort_result.append(-1)  # High volume, low result = weakness
                elif not vol_high and price_move > range_size * 0.7:
                    effort_result.append(-1)  # Low volume, high result = unsustainable
                else:
                    effort_result.append(1)   # Normal
            
            state.effort_result_score = sum(effort_result) / len(effort_result) * 100
            
            # Save metrics
            df.to_csv(get_path("metrics.csv"), index=False)
            
            print(f"✅ Laws: SD={state.supply_demand_score:.1f}% | Range={state.cause_effect_range:.2f} | ER={state.effort_result_score:.1f}%")
            return f"STATE:{state.to_json()}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))


# =============================================================================
# NEW TOOL: INTRADAY SUPPLY/DEMAND ANALYZER
# =============================================================================
class IntradaySupplyDemandTool(BaseTool):
    name: str = "Intraday Analyzer"
    description: str = "Analyzes intraday volume-price relationships to detect institutional activity"
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            
            # Fetch intraday data
            intraday_df = self._fetch_intraday_data(
                start_date=state.current_date.replace('-', ''),
                days_before=10
            )
            
            if intraday_df.empty:
                print("⚠️ No intraday data available")
                state.intraday_records = 0
                return f"STATE:{state.to_json()}"
            
            state.intraday_records = len(intraday_df)
            
            # Sort and prepare data
            intraday_df = intraday_df.sort_values(['DATE', 'price']).reset_index(drop=True)
            intraday_df['price_change'] = intraday_df['price'].diff()
            intraday_df['price_change_pct'] = (intraday_df['price'].pct_change() * 100).fillna(0)
            
            # Volume metrics
            vol_mean = intraday_df['totalVolume'].mean()
            vol_std = intraday_df['totalVolume'].std()
            intraday_df['volume_zscore'] = (intraday_df['totalVolume'] - vol_mean) / vol_std if vol_std > 0 else 0
            
            # Detect Wyckoff patterns
            patterns = self._detect_patterns(intraday_df)
            state.climactic_volume_events = patterns['climactic_volume']
            state.no_demand_bars = patterns['no_demand']
            state.no_supply_bars = patterns['no_supply']
            state.stopping_volume_bars = patterns['stopping_volume']
            state.intraday_institutional_signals = sum(patterns.values())
            
            # Absorption analysis
            high_vol_bars = intraday_df[intraday_df['volume_zscore'] > 1.5]
            if len(high_vol_bars) > 0:
                absorption_bars = high_vol_bars[abs(high_vol_bars['price_change_pct']) < 0.5]
                state.intraday_absorption_score = (len(absorption_bars) / len(high_vol_bars)) * 100
            else:
                state.intraday_absorption_score = 0.0
            
            # Supply/Demand dominance
            intraday_df['vol_weighted_change'] = intraday_df['price_change'] * intraday_df['totalVolume']
            total_vol_weighted = intraday_df['vol_weighted_change'].sum()
            total_volume = intraday_df['totalVolume'].sum()
            net_pressure = total_vol_weighted / total_volume if total_volume > 0 else 0
            
            if net_pressure > 0.02:
                state.intraday_dominant_force = "DEMAND"
            elif net_pressure < -0.02:
                state.intraday_dominant_force = "SUPPLY"
            else:
                state.intraday_dominant_force = "NEUTRAL"
            
            # Save files
            intraday_df.to_csv(get_path("intraday_analysis.csv"), index=False)
            
            summary = {
                "total_records": state.intraday_records,
                "dominant_force": state.intraday_dominant_force,
                "absorption_score": round(state.intraday_absorption_score, 2),
                "institutional_signals": state.intraday_institutional_signals,
                "patterns": {
                    "climactic_volume": state.climactic_volume_events,
                    "no_demand": state.no_demand_bars,
                    "no_supply": state.no_supply_bars,
                    "stopping_volume": state.stopping_volume_bars
                }
            }
            
            with open(get_path("intraday_summary.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Intraday: {state.intraday_records} records | Force={state.intraday_dominant_force} | Absorption={state.intraday_absorption_score:.1f}%")
            
            return f"STATE:{state.to_json()}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))
    
    def _detect_patterns(self, df):
        """Detect Wyckoff intraday patterns"""
        patterns = {'climactic_volume': 0, 'no_demand': 0, 'no_supply': 0, 'stopping_volume': 0}
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            price_change = row['price_change']
            price_change_pct = abs(row['price_change_pct'])
            vol_zscore = row['volume_zscore']
            
            # Climactic Volume: High volume, small price movement
            if vol_zscore > 2.0 and price_change_pct < 0.5:
                patterns['climactic_volume'] += 1
            
            # No Demand: Price rises, low volume
            elif price_change > 0 and vol_zscore < -1.0:
                patterns['no_demand'] += 1
            
            # No Supply: Price falls, low volume
            elif price_change < 0 and vol_zscore < -1.0:
                patterns['no_supply'] += 1
            
            # Stopping Volume: Extremely high volume, minimal movement
            elif vol_zscore > 2.5 and price_change_pct < 0.3:
                patterns['stopping_volume'] += 1
        
        return patterns
    
    def _fetch_intraday_data(self, start_date: str, days_before=10):
        """Fetch intraday data from cafef.vn"""
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        data_start_date = start_dt - timedelta(days=days_before-1)
        
        date_list = []
        current_date = data_start_date
        while current_date <= start_dt:
            date_list.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        
        all_data = []
        url = "https://msh-appdata.cafef.vn/rest-api/api/v1/MatchPrice"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Origin": "https://cafef.vn",
            "Referer": "https://cafef.vn/"
        }
        
        for intraday_date in date_list:
            try:
                params = {"symbol": "HPG", "date": intraday_date}
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                match_data = data.get('aggregates')
                
                if match_data:
                    df_day = pd.DataFrame(match_data)
                    df_day['DATE'] = pd.to_datetime(intraday_date, format='%Y%m%d').date()
                    all_data.append(df_day)
            except Exception as e:
                print(f"⚠️ Skipped {intraday_date}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


# =============================================================================
# TOOLS - ORIGINAL (UNCHANGED)
# =============================================================================
class WyckoffEventDetector(BaseTool):
    name: str = "Event Detector"
    description: str = "Detects Wyckoff events: SC, AR, Spring, SOS, UTAD, SOW, etc."
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            df = pd.read_csv(get_path("metrics.csv"))
            df['Date'] = pd.to_datetime(df['Date'])
            
            events = []
            
            for i in range(30, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-30:i]
                date_str = row['Date'].strftime('%Y-%m-%d')
                price = row['Close']
                
                # SELLING CLIMAX (SC) - High volume, wide spread, closes off lows
                if (row['Volume_Ratio'] > 2.0 and 
                    row['High_Low_Range'] > prev['High_Low_Range'].mean() * 1.5):
                    close_pos = (row['Close'] - row['Low']) / row['High_Low_Range']
                    if close_pos > 0.4:  # Closes in upper half
                        events.append(f"{date_str}|SC|{price:.2f}")
                
                # AUTOMATIC RALLY (AR) - Strong bounce after SC
                if (len(events) > 0 and 'SC' in events[-1] and 
                    row['Price_Change'] > prev['High_Low_Range'].mean()):
                    events.append(f"{date_str}|AR|{price:.2f}")
                
                # SPRING - Price breaks support then recovers with volume
                support = prev['Low'].min()
                if (row['Low'] < support * 0.985 and 
                    row['Close'] > support and 
                    row['Volume_Ratio'] > 1.3):
                    events.append(f"{date_str}|SPRING|{price:.2f}")
                
                # SIGN OF STRENGTH (SOS) - Price breaks resistance with volume
                resistance = prev['High'].max()
                if (row['Close'] > resistance and 
                    row['Volume_Ratio'] > 1.5):
                    events.append(f"{date_str}|SOS|{price:.2f}")
                
                # UPTHRUST AFTER DISTRIBUTION (UTAD) - False breakout
                if (row['High'] > resistance and 
                    row['Close'] < resistance and 
                    row['Volume_Ratio'] > 1.5):
                    events.append(f"{date_str}|UTAD|{price:.2f}")
                
                # SIGN OF WEAKNESS (SOW)
                if (row['Price_Change'] < -prev['High_Low_Range'].mean() * 0.5 and 
                    row['Volume_Ratio'] > 1.5):
                    events.append(f"{date_str}|SOW|{price:.2f}")
            
            state.event_count = len(events)
            
            # Save events to file (ALWAYS create file, even if empty)
            events_file = get_path("events.txt")
            with open(events_file, "w", encoding="utf-8") as f:
                if events:
                    f.write("\n".join(events))
                else:
                    f.write("# No events detected yet\n")
            
            # Verify file was created
            if not os.path.exists(events_file):
                raise WyckoffError(self.name, ErrorSeverity.CRITICAL, "Failed to save events file")
            
            print(f"✅ Detected {len(events)} events")
            if events:
                print(f"   Latest: {events[-1]}")
            
            return f"STATE:{state.to_json()}\nEVENTS_FILE:{events_file}\nCOUNT:{len(events)}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))


class WyckoffPhaseIdentifier(BaseTool):
    name: str = "Phase Identifier"
    description: str = "Identifies Wyckoff schematic and phase"
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            
            # Read events directly from file (more reliable)
            events_file = get_path("events.txt")
            event_types = []
            
            if os.path.exists(events_file):
                with open(events_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 2:
                                event_types.append(parts[1])
            else:
                print(f"⚠️ Events file not found, will use fallback analysis")
            
            # Analyze recent events (last 10)
            recent_events = event_types[-10:] if event_types else []
            has_spring = 'SPRING' in recent_events
            has_sos = 'SOS' in recent_events
            has_utad = 'UTAD' in recent_events
            has_sow = 'SOW' in recent_events
            
            # Determine schematic based on events
            if has_sos or has_spring:
                state.schematic = "ACCUMULATION"
                if has_sos:
                    state.phase = "PHASE D/E - Markup Beginning"
                elif has_spring:
                    state.phase = "PHASE C - Spring Detected"
                else:
                    state.phase = "PHASE B - Building Cause"
            elif has_utad or has_sow:
                state.schematic = "DISTRIBUTION"
                if has_sow:
                    state.phase = "PHASE D/E - Markdown Beginning"
                elif has_utad:
                    state.phase = "PHASE C - UTAD Detected"
                else:
                    state.phase = "PHASE B - Building Cause"
            else:
                # Use price position in range (with zero-division protection)
                current = state.current_price
                range_size = state.resistance - state.support
                
                if range_size > 0:  # Only calculate if we have a valid range
                    range_pos = (current - state.support) / range_size
                    
                    if range_pos > 0.7:
                        state.schematic = "DISTRIBUTION"
                        state.phase = "PHASE B/C - Upper Range"
                    elif range_pos < 0.3:
                        state.schematic = "ACCUMULATION"
                        state.phase = "PHASE B/C - Lower Range"
                    else:
                        state.schematic = "NEUTRAL"
                        state.phase = "PHASE B - Mid Range"
                else:
                    # No clear range established yet
                    state.schematic = "NEUTRAL"
                    state.phase = "PHASE A - Early Development"
            
            # NEW: Intraday confirmation (if available)
            confirmations = []
            if state.intraday_records > 0:
                if state.intraday_absorption_score > 50:
                    confirmations.append(f"High absorption ({state.intraday_absorption_score:.0f}%) detected")
                
                if state.intraday_dominant_force == "DEMAND" and state.schematic == "ACCUMULATION":
                    confirmations.append("Intraday demand confirms accumulation")
                elif state.intraday_dominant_force == "SUPPLY" and state.schematic == "DISTRIBUTION":
                    confirmations.append("Intraday supply confirms distribution")
                
                if state.no_supply_bars > 5 and state.schematic == "ACCUMULATION":
                    confirmations.append(f"No supply bars ({state.no_supply_bars}) support bullish setup")
            
            print(f"✅ {state.schematic} | {state.phase}")
            if confirmations:
                print(f"   Intraday confirmations: {len(confirmations)}")
            
            # Save phase analysis summary to file for debugging
            phase_file = get_path("phase_analysis.txt")
            with open(phase_file, 'w', encoding='utf-8') as f:
                f.write(f"Schematic: {state.schematic}\n")
                f.write(f"Phase: {state.phase}\n")
                f.write(f"Support: {state.support:.2f}\n")
                f.write(f"Resistance: {state.resistance:.2f}\n")
                f.write(f"Events analyzed: {len(recent_events)}\n")
                if confirmations:
                    f.write(f"\nIntraday Confirmations:\n")
                    for conf in confirmations:
                        f.write(f"  - {conf}\n")
            
            return f"STATE:{state.to_json()}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))


class PriceTargetCalculator(BaseTool):
    name: str = "Target Calculator"
    description: str = "Calculates entry, stop loss, and price targets"
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            
            current = state.current_price
            range_size = state.cause_effect_range
            
            # Entry zone (tight around current)
            state.entry_low = current * 0.98
            state.entry_high = current * 1.02
            
            # Stop loss (below support with buffer)
            state.stop_loss = state.support * 0.97 if state.support > 0 else current * 0.92
            
            # Targets based on Wyckoff Cause & Effect
            if "ACCUMULATION" in state.schematic:
                # Bullish targets
                state.target1 = current + (range_size * 0.382)  # Conservative
                state.target2 = current + (range_size * 0.618)  # Fibonacci extension
            else:
                # Bearish or neutral - lower targets
                state.target1 = current + (range_size * 0.236)
                state.target2 = current + (range_size * 0.382)
            
            risk_reward = (state.target1 - current) / (current - state.stop_loss)
            
            print(f"✅ Entry: {state.entry_low:.2f}-{state.entry_high:.2f} | R:R = {risk_reward:.2f}")
            return f"STATE:{state.to_json()}"
            
        except Exception as e:
            raise WyckoffError(self.name, ErrorSeverity.CRITICAL, str(e))


class PriceValidator(BaseTool):
    name: str = "Price Validator"
    description: str = "Validates price levels are logical"
    
    def _run(self, state_json: str) -> str:
        try:
            state = WyckoffState.from_json(state_json)
            errors = []
            
            if state.stop_loss >= state.current_price:
                errors.append("Stop loss above current price")
            
            if state.target1 <= state.current_price:
                errors.append("Target 1 below current price")
            
            if state.entry_high < state.entry_low:
                errors.append("Entry range inverted")
            
            if errors:
                return f"FAIL: {', '.join(errors)}"
            
            print(f"✅ Validation passed")
            return f"PASS\nSTATE:{state.to_json()}"
            
        except Exception as e:
            return f"FAIL: {str(e)}"


# =============================================================================
# LLM SETUP
# =============================================================================
ollama_llm = LLM(model="ollama/deepseek-v3.1:671b-cloud", is_litellm=True)

# =============================================================================
# AGENTS - ORIGINAL + NEW INTRADAY AGENT
# =============================================================================
data_agent = Agent(
    role='Data Collector',
    goal='Fetch HPG and VNINDEX data',
    backstory='Expert in fetching Vietnamese stock data. Returns STATE JSON.',
    tools=[WyckoffDataTool()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

law_analyst = Agent(
    role='Wyckoff Laws Analyst',
    goal='Analyze 3 Wyckoff Laws',
    backstory="""Master of Wyckoff Laws:
    - Law 1: Supply & Demand balance
    - Law 2: Cause (range) predicts Effect (move)
    - Law 3: Effort (volume) matches Result (price)
    
    CRITICAL: You MUST use the Law Analyzer tool.
    Extract the STATE JSON from previous task output.
    Look for the text "STATE:" followed by a JSON object.
    Pass that entire JSON string to the tool's state_json parameter.""",
    tools=[WyckoffLawAnalyzer()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

# NEW: Intraday agent
intraday_agent = Agent(
    role='Intraday Supply/Demand Specialist',
    goal='Detect institutional activity through intraday volume-price analysis',
    backstory="""Expert in Wyckoff intraday analysis focusing on:
    - Effort (volume) vs Result (price movement)
    - Institutional footprints (absorption, climactic volume)
    - Supply/Demand imbalances
    
    CRITICAL: You MUST use the Intraday Analyzer tool.
    Extract STATE JSON from previous output and pass to tool.
    
    DO NOT interpret patterns - only report detected metrics.""",
    tools=[IntradaySupplyDemandTool()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

event_detector = Agent(
    role='Event Detector',
    goal='Detect Wyckoff events',
    backstory="""Specialist in spotting SC, Spring, SOS, UTAD, SOW.
    
    CRITICAL: You MUST use the Event Detector tool.
    Extract STATE JSON from previous output (line starting with "STATE:") and pass to tool.
    The tool will detect events and save them to file.""",
    tools=[WyckoffEventDetector()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

phase_analyst = Agent(
    role='Phase Analyst',
    goal='Identify schematic and phase',
    backstory="""Expert in Wyckoff schematics (Accumulation/Distribution) and phases (A-E).
    
    Extract STATE JSON from previous output (line starting with "STATE:") and pass to tool.
    The tool reads events from file automatically.""",
    tools=[WyckoffPhaseIdentifier()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

price_calculator = Agent(
    role='Price Calculator',
    goal='Calculate trading levels',
    backstory="""Calculates entry, stops, targets based on Wyckoff principles.
    
    Extract STATE JSON from previous output and pass to tool.""",
    tools=[PriceTargetCalculator()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

validator = Agent(
    role='Validator',
    goal='Validate price levels',
    backstory="""Ensures all price levels are logical and tradeable.
    
    Extract STATE JSON from previous output and pass to tool.""",
    tools=[PriceValidator()],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

reporter = Agent(
    role='Vietnamese Report Writer',
    goal='Write comprehensive Vietnamese analysis report',
    backstory=f"""Chuyên gia viết báo cáo Wyckoff bằng tiếng Việt.
    
    Đọc STATE JSON để lấy tất cả thông tin:
    - Giá hiện tại, ngày
    - Schematic và Phase
    - Support/Resistance
    - 3 Laws scores
    - Entry/Stop/Targets
    - Intraday metrics (absorption, dominant force, patterns)
    
    Đọc events từ: {get_path("events.txt")}
    Đọc intraday summary từ: {get_path("intraday_summary.json")}
    
    Viết báo cáo chi tiết, chuyên nghiệp.""",
    tools=[],
    llm=ollama_llm,
    verbose=False,
    allow_delegation=False
)

# =============================================================================
# TASKS - ORIGINAL + NEW INTRADAY TASK
# =============================================================================
task_data = Task(
    description="Use Data Fetcher tool to get HPG + VNINDEX. Return STATE JSON.",
    expected_output="STATE JSON with price, date, data points",
    agent=data_agent,
    output_file=get_path("01_data.txt")
)

task_laws = Task(
    description="""CRITICAL: You MUST use the Law Analyzer tool.
    
    1. Extract STATE JSON from previous output (look for "STATE:{...}").
    2. Call the Law Analyzer tool with state_json parameter.
    3. The tool will analyze all 3 Wyckoff Laws and return updated STATE.""",
    expected_output="STATE JSON with law scores (supply_demand_score, cause_effect_range, effort_result_score)",
    agent=law_analyst,
    context=[task_data],
    output_file=get_path("02_laws.txt")
)

# NEW: Intraday task (inserted after laws, before events)
task_intraday = Task(
    description="""CRITICAL: You MUST use the Intraday Analyzer tool.
    
    1. Extract STATE JSON from previous law analysis output.
    2. Call Intraday Analyzer tool with state_json parameter.
    3. Tool will detect institutional patterns through volume-price analysis.
    
    DO NOT interpret patterns - only report metrics detected.""",
    expected_output="STATE JSON with intraday metrics (absorption_score, dominant_force, pattern counts)",
    agent=intraday_agent,
    context=[task_laws],
    output_file=get_path("02b_intraday.txt")
)

task_events = Task(
    description="""CRITICAL: You MUST use the Event Detector tool.
    
    1. Extract STATE JSON from previous output.
    2. Call Event Detector tool with state_json parameter.
    3. Tool will detect SC, AR, Spring, SOS, UTAD, SOW events.""",
    expected_output="STATE JSON with event_count, events saved to file",
    agent=event_detector,
    context=[task_intraday],  # Updated to use intraday context
    output_file=get_path("03_events.txt")
)

task_phase = Task(
    description="""CRITICAL: You MUST use the Phase Identifier tool.
    
    1. Extract STATE JSON from previous output.
    2. Call Phase Identifier tool with state_json parameter.
    3. Tool determines schematic (Accumulation/Distribution) and phase (A-E).""",
    expected_output="STATE JSON with schematic and phase identified",
    agent=phase_analyst,
    context=[task_events],
    output_file=get_path("04_phase.txt")
)

task_targets = Task(
    description="""Extract STATE JSON from previous output.
    Use Target Calculator tool for entry/stop/targets.""",
    expected_output="STATE JSON with price levels",
    agent=price_calculator,
    context=[task_phase],
    output_file=get_path("05_targets.txt")
)

task_validate = Task(
    description="""Extract STATE JSON from previous output.
    Use Price Validator tool to check logic.""",
    expected_output="PASS or FAIL with STATE JSON",
    agent=validator,
    context=[task_targets],
    output_file=get_path("06_validation.txt")
)

task_report = Task(
    description=f"""Viết báo cáo Wyckoff đầy đủ bằng tiếng Việt (600-700 từ).

Extract STATE JSON từ các task trước để lấy:
- current_price, current_date
- schematic, phase  
- support, resistance, cause_effect_range
- supply_demand_score, effort_result_score
- entry_low, entry_high, stop_loss, target1, target2
- Intraday: absorption_score, dominant_force, climactic_volume_events, no_demand_bars, no_supply_bars, stopping_volume_bars

Đọc tất cả events từ file: {get_path("events.txt")}
Đọc intraday summary từ: {get_path("intraday_summary.json")}

CẤU TRÚC BÁO CÁO:

# PHÂN TÍCH WYCKOFF - HPG
Ngày: [current_date]

## I. TỔNG QUAN THỊ TRƯỜNG
- Giá hiện tại: [current_price] VND
- Schematic: [schematic]
- Giai đoạn: [phase]

## II. PHÂN TÍCH 3 ĐỊNH LUẬT WYCKOFF

### Định Luật 1: Cung & Cầu
Score: [supply_demand_score]%
Giải thích: [Dương = cầu mạnh, Âm = cung mạnh]

### Định Luật 2: Nguyên Nhân & Kết Quả
Trading Range: [cause_effect_range] VND
- Support: [support]
- Resistance: [resistance]
Dự báo: [Range càng rộng, move càng mạnh]

### Định Luật 3: Nỗ Lực & Kết Quả
Score: [effort_result_score]%
Giải thích: [Volume có khớp với price không]

## III. PHÂN TÍCH INTRADAY - HOẠT ĐỘNG TỔ CHỨC

### A. Tổng Quan
- Số lượng bars: [intraday_records]
- Lực lượng chi phối: [intraday_dominant_force]
- Tín hiệu tổ chức: [intraday_institutional_signals]

### B. Nỗ Lực vs Kết Quả (Intraday)
- Absorption Score: [intraday_absorption_score]%
  * >60% = Hấp thụ mạnh (tổ chức tích lũy/phân phối)
  * 30-60% = Hấp thụ vừa
  * <30% = Ít hấp thụ

### C. Các Pattern Phát Hiện
- Climactic Volume: [climactic_volume_events] sự kiện
  → Volume cao, giá ít biến động = Hấp thụ
- Stopping Volume: [stopping_volume_bars] bars
  → Volume cực cao, xu hướng kiệt sức
- No Supply: [no_supply_bars] bars
  → Giá giảm, volume thấp = áp lực bán yếu
- No Demand: [no_demand_bars] bars
  → Giá tăng, volume thấp = áp lực mua yếu

## IV. SỰ KIỆN WYCKOFF QUAN TRỌNG (DAILY)
Liệt kê 5-7 sự kiện gần nhất từ events.txt với giải thích

## V. KHUYẾN NGHỊ GIAO DỊCH

**Vùng Mua Vào:** [entry_low] - [entry_high] VND
**Điểm Cắt Lỗ:** [stop_loss] VND  
**Mục Tiêu 1:** [target1] VND (Conservative)
**Mục Tiêu 2:** [target2] VND (Aggressive)

**Tỷ lệ Risk:Reward:** [tính toán]

## VI. KẾT LUẬN & LƯU Ý
- Tổng hợp quan điểm
- Độ tin cậy (dựa vào daily + intraday confirmation)
- Điều kiện setup hợp lệ

---
*Phân tích kết hợp Daily + Intraday theo phương pháp Wyckoff*""",
    expected_output="Báo cáo tiếng Việt đầy đủ với tất cả số liệu từ STATE",
    agent=reporter,
    context=[task_validate],
    output_file=get_path("FINAL_REPORT.md")
)

# =============================================================================
# CREW - UPDATED WITH INTRADAY
# =============================================================================
wyckoff_crew = Crew(
    agents=[data_agent, law_analyst, intraday_agent, event_detector, phase_analyst, 
            price_calculator, validator, reporter],
    tasks=[task_data, task_laws, task_intraday, task_events, task_phase, 
           task_targets, task_validate, task_report],
    verbose=True,
    process="sequential"
)

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("🚀 WYCKOFF METHOD ANALYSIS - WITH INTRADAY INSTITUTIONAL DETECTION")
    print("="*80)
    print(f"📁 Output directory: {WORK_DIR}")
    print("\n✨ Features:")
    print("  • File-based state passing (token efficient)")
    print("  • Complete Wyckoff Laws analysis (Daily)")
    print("  • Intraday institutional activity detection (NEW)")
    print("  • Comprehensive event detection (SC, Spring, SOS, UTAD, etc.)")
    print("  • Phase & Schematic identification")
    print("  • Price targets with validation")
    print("  • Vietnamese professional report")
    print("="*80)
    
    try:
        result = wyckoff_crew.kickoff()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        
        # File verification
        print("\n📊 Generated Files:")
        files = {
            "data.csv": "Raw OHLCV + Index",
            "metrics.csv": "Calculated metrics",
            "intraday_analysis.csv": "Intraday data with patterns (NEW)",
            "intraday_summary.json": "Intraday metrics summary (NEW)",
            "events.txt": "Wyckoff events",
            "01_data.txt": "Data collection log",
            "02_laws.txt": "Laws analysis log",
            "02b_intraday.txt": "Intraday analysis log (NEW)",
            "03_events.txt": "Event detection log",
            "04_phase.txt": "Phase identification log",
            "05_targets.txt": "Price targets log",
            "06_validation.txt": "Validation log",
            "FINAL_REPORT.md": "⭐ Vietnamese Report"
        }
        
        for filename, desc in files.items():
            path = get_path(filename)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✅ {filename:25s} ({size:6,} bytes) - {desc}")
            else:
                print(f"  ❌ {filename:25s} - NOT FOUND")
        
        print(f"\n📖 Read final report: {get_path('FINAL_REPORT.md')}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()