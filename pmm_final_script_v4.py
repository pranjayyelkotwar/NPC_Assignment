import logging
from decimal import Decimal
from typing import Dict, List
import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class PMMFullStrategy2(ScriptStrategyBase):
    """
    BotCamp Module 3 - Market Making Strategies
    Description:
    A comprehensive PMM strategy combining static spreads, volatility-adjusted spreads (NATR),
    trend-based price shifts (RSI), and inventory-based price shifts.
    Enhanced with multi-timeframe analysis, volume profile, and market sentiment.
    """
    # Initial static spreads (fallback if dynamic calculation fails)
    bid_spread = 0.0025
    ask_spread = 0.0025
    order_refresh_time = 90  # Seconds between order refreshes
    order_amount = 1 # Amount per order
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice  # Use mid-price as base reference
    base , quote = trading_pair.split("-")
    # Candles params
    candle_exchange = "binance"
    candles_interval = "1m"  # High-frequency candles for responsiveness
    candles_length = 14  # Lookback period for indicators
    max_records = 1000  # Max candle history

    # Volatility spread params
    bid_spread_scalar = 1  # Multiplier for bid spread based on NATR
    ask_spread_scalar = 0.8   # Multiplier for ask spread based on NATR

    # Price shift params
    max_shift_spread = 0.0005  # 50 bps
    orig_price = 1
    reference_price = 1
    price_multiplier = 0  # Trend-based multiplier
    trend_scalar = -1  # Negative to shift down in overbought conditions

    # Inventory params
    target_ratio = 0.5  # Target base:total asset ratio cuz 
    current_ratio = 0.5  # Current ratio
    inventory_delta = 0  # Difference from target
    inventory_scalar = 3  # Multiplier for inventory adjustment
    inventory_multiplier = 0  # Inventory-based multiplier

    # Multi-timeframe parameters
    timeframes = ["1m", "15m", "1h"]
    timeframe_weights = {"1m": 0.5, "15m": 0.3, "1h": 0.2}  # Sum should be 1
    
    # Volume profile parameters
    volume_profile_lookback = 24  # hours to look back for volume profile
    volume_profile_bins = 20  # number of price bins
    high_volume_levels = {}  # Will store high volume price levels
    
    # Market sentiment parameter (hardcoded to 0.75 negative for ETH-USDT)
    market_sentiment = -0.75  # Range from -1 (very negative) to 1 (very positive)
    sentiment_impact_factor = 0.0003  # How much sentiment affects spreads (30 bps max)
    
    # Initialize main candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))

    # Define markets
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.logger().setLevel(logging.INFO)  # Ensure logging is enabled
        self.candles.start()
        
        # Initialize additional timeframe candles
        self.candles_dict = {
            "1m": self.candles  # Use the main candles for 1m
        }
        
        # Create candles for additional timeframes
        for tf in self.timeframes:
            if tf != "1m":  # Already have 1m
                self.candles_dict[tf] = CandlesFactory.get_candle(CandlesConfig(
                    connector=self.candle_exchange,
                    trading_pair=self.trading_pair,
                    interval=tf,
                    max_records=self.max_records
                ))
                self.candles_dict[tf].start()
                
        self.log_with_clock(logging.INFO, f"Strategy initialized with multi-timeframe analysis. Timeframes: {', '.join(self.timeframes)}")
        self.log_with_clock(logging.INFO, f"Market sentiment set to {self.market_sentiment} for {self.trading_pair}")

    # Replace fixed order_amount with volatility-based sizing
    def get_order_amount(self):
        natr = self.get_candles_with_features()[f"NATR_{self.candles_length}"].iloc[-1]
        base_amount = self.order_amount
        volatility_multiplier = 1 / (1 + natr * 10)  # Reduce size in high volatility
        return base_amount * volatility_multiplier

    def on_stop(self):
        self.candles.stop()
        # Stop all additional timeframe candles
        for tf, candle in self.candles_dict.items():
            if tf != "1m":  # 1m already stopped
                candle.stop()
        self.log_with_clock(logging.INFO, "Strategy stopped. All candles feeds terminated.")

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            # self.log_with_clock(logging.INFO, "Tick triggered. Starting order cycle.")
            self.cancel_all_orders()
            # Calculate volume profile before updating multipliers
            self.calculate_volume_profile()
            self.update_multipliers()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
            # self.log_with_clock(logging.INFO, f"Order cycle completed. Next refresh in {self.order_refresh_time} seconds.")

    def get_candles_with_features(self):
        candles_df = self.candles.candles_df
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
        candles_df['bid_spread_bps'] = candles_df[f"NATR_{self.candles_length}"] * self.bid_spread_scalar * 10000
        candles_df['ask_spread_bps'] = candles_df[f"NATR_{self.candles_length}"] * self.ask_spread_scalar * 10000
        candles_df.ta.rsi(length=self.candles_length, append=True)
        # self.log_with_clock(logging.INFO, "Candles updated with NATR and RSI features.")
        return candles_df
    
    def get_multi_timeframe_rsi(self):
        """Calculate RSI across multiple timeframes and return weighted average"""
        weighted_rsi = 0
        
        for tf, weight in self.timeframe_weights.items():
            if tf in self.candles_dict and not self.candles_dict[tf].candles_df.empty:
                df = self.candles_dict[tf].candles_df.copy()
                df.ta.rsi(length=self.candles_length, append=True)
                
                # Ensure we have RSI values before accessing
                if f"RSI_{self.candles_length}" in df.columns and not df[f"RSI_{self.candles_length}"].empty:
                    rsi_value = df[f"RSI_{self.candles_length}"].iloc[-1]
                    if not pd.isna(rsi_value):
                        weighted_rsi += rsi_value * weight
                    else:
                        weighted_rsi += 50 * weight  # Default to neutral if NaN
                else:
                    weighted_rsi += 50 * weight  # Default to neutral if column doesn't exist
        
        self.log_with_clock(logging.INFO, f"Multi-timeframe weighted RSI: {weighted_rsi:.2f}")
        return weighted_rsi
    
    def calculate_volume_profile(self):
        """Calculate volume profile to find support/resistance levels"""
        # Use 1h candles for volume profile to capture more meaningful data
        if "1h" in self.candles_dict and not self.candles_dict["1h"].candles_df.empty:
            df = self.candles_dict["1h"].candles_df.tail(self.volume_profile_lookback)
            
            if not df.empty:
                # Create price bins
                price_min = df['low'].min()
                price_max = df['high'].max()
                
                if price_min is not None and price_max is not None and price_min < price_max:
                    bin_size = (price_max - price_min) / self.volume_profile_bins
                    
                    # Group volumes by price level
                    df['price_bin'] = ((df['close'] - price_min) / bin_size).astype(int)
                    volume_profile = df.groupby('price_bin')['volume'].sum()
                    
                    # Find high volume nodes (above average)
                    high_volume_nodes = volume_profile[volume_profile > volume_profile.mean()]
                    
                    # Convert bin indices back to prices
                    self.high_volume_levels = {
                        (price_min + (bin_idx * bin_size)): vol 
                        for bin_idx, vol in high_volume_nodes.items()
                    }
                    
                    self.log_with_clock(logging.INFO, f"Volume profile calculated. Found {len(self.high_volume_levels)} high volume levels.")
        else:
            self.log_with_clock(logging.INFO, "1h candles not available or empty. Skipping volume profile calculation.")
    
    def get_market_sentiment(self):
        """Get market sentiment (hardcoded for ETH-USDT as requested)"""
        # Currently hardcoded as -0.75 (negative sentiment for ETH-USDT)
        self.log_with_clock(logging.INFO, f"Market sentiment: {self.market_sentiment}")
        return self.market_sentiment
    
    def find_nearest_volume_level(self, price):
        """Find the nearest high volume level to the given price"""
        if not self.high_volume_levels:
            return None
            
        nearest_level = None
        min_distance = float('inf')
        
        for level in self.high_volume_levels:
            distance = abs(level - price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
                
        return nearest_level

    def update_multipliers(self):
        candles_df = self.get_candles_with_features()
        
        # Volatility-based spreads
        natr = candles_df[f"NATR_{self.candles_length}"].iloc[-1]
        self.bid_spread = natr * self.bid_spread_scalar if natr else self.bid_spread
        self.ask_spread = natr * self.ask_spread_scalar if natr else self.ask_spread
        self.log_with_clock(logging.INFO, f"Spreads updated: Bid={self.bid_spread:.6f}, Ask={self.ask_spread:.6f} (NATR={natr:.6f})")

        self.bid_spread = max(natr * self.bid_spread_scalar, 0.0005)  # Min 20 bps
        self.ask_spread = max(natr * self.ask_spread_scalar, 0.0005)  # Min 20 bps
        
        # Get multi-timeframe RSI for trend-based price shift
        multi_tf_rsi = self.get_multi_timeframe_rsi()
        self.price_multiplier = (multi_tf_rsi - 50) / 50 * self.max_shift_spread * self.trend_scalar if multi_tf_rsi else 0
        self.log_with_clock(logging.INFO, f"Multi-timeframe trend shift: RSI={multi_tf_rsi:.2f}, Price Multiplier={self.price_multiplier:.8f}")

        # Inventory-based price shift
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        total_bal = base_bal_in_quote + quote_bal
        self.current_ratio = float(base_bal_in_quote / total_bal) if total_bal > 0 else 0.5
        delta = (self.target_ratio - self.current_ratio) / self.target_ratio if self.target_ratio != 0 else 0
        self.inventory_delta = max(-1, min(1, delta))
        self.inventory_multiplier = self.inventory_delta * self.max_shift_spread * self.inventory_scalar
        self.log_with_clock(logging.INFO, f"Inventory shift: Current Ratio={self.current_ratio:.4f}, Delta={self.inventory_delta:.4f}, Inventory Multiplier={self.inventory_multiplier:.8f}")
        
        # Add sentiment adjustment
        sentiment = self.get_market_sentiment()
        sentiment_adjustment = sentiment * self.sentiment_impact_factor
        self.log_with_clock(logging.INFO, f"Sentiment adjustment: {sentiment_adjustment:.8f}")
        
        # Calculate shifted reference price with all factors
        self.orig_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        self.reference_price = self.orig_price * Decimal(str(1 + self.price_multiplier)) * \
                               Decimal(str(1 + self.inventory_multiplier)) * \
                               Decimal(str(1 + sentiment_adjustment))
        
        self.log_with_clock(logging.INFO, f"Reference price updated: Original={self.orig_price:.4f}, Shifted={self.reference_price:.4f}")

    def create_proposal(self) -> List[OrderCandidate]:
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        mid_price = (best_ask + best_bid) / 2
        
        buy_price = self.reference_price * Decimal(1 - self.bid_spread)
        sell_price = self.reference_price * Decimal(1 + self.ask_spread)
        
        # Check if there are high volume levels nearby that we should target
        nearest_level_to_bid = self.find_nearest_volume_level(float(buy_price))
        nearest_level_to_ask = self.find_nearest_volume_level(float(sell_price))
        
        # If a high volume level is within 0.5% of our calculated price, target that level instead
        if nearest_level_to_bid and abs(nearest_level_to_bid - float(buy_price)) / float(buy_price) < 0.005:
            buy_price = Decimal(str(nearest_level_to_bid))
            self.log_with_clock(logging.INFO, f"Adjusted buy price to target high volume level: {buy_price:.4f}")
            
        if nearest_level_to_ask and abs(nearest_level_to_ask - float(sell_price)) / float(sell_price) < 0.005:
            sell_price = Decimal(str(nearest_level_to_ask))
            self.log_with_clock(logging.INFO, f"Adjusted sell price to target high volume level: {sell_price:.4f}")
        
        final_buy_price = max(buy_price, best_bid * Decimal("0.998"))  # 0.2% inside bid
        final_sell_price = min(sell_price, best_ask * Decimal("1.002"))  # 0.2% inside ask
        
        self.log_with_clock(logging.INFO, f"Proposal created: Buy Price={final_buy_price:.4f} (Best Bid={best_bid:.4f}), Mid Price={mid_price:.4f}, Sell Price={final_sell_price:.4f} (Best Ask={best_ask:.4f})")
        self.order_amount = self.get_order_amount()

        buy_order = OrderCandidate(
            trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            order_side=TradeType.BUY, amount=Decimal(self.order_amount), price=final_buy_price
        )
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            order_side=TradeType.SELL, amount=Decimal(self.order_amount), price=final_sell_price
        )
        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        # self.log_with_clock(logging.INFO, "Proposal adjusted to budget constraints.")
        return adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(self.exchange, order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        self.log_with_clock(logging.INFO, f"Placing order: {order.order_side.name} {order.amount} {order.trading_pair} at {order.price:.4f}")
        if order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        # self.log_with_clock(logging.INFO, "All active orders canceled.")

    def did_fill_order(self, event: OrderFilledEvent):
        # Calculate trade fee
        trade_fee = event.trade_fee.percent if event.trade_fee else 0.001  # Default to 0.1% if no fee data
        fee_amount = event.price * event.amount * trade_fee
        profit = (event.price - fee_amount) if event.trade_type == TradeType.SELL else -(event.price + fee_amount)
        msg = f"Order filled: {event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} at {round(event.price, 4)}, Profit={profit:.4f}, Fee={fee_amount:.4f}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        # Balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        # Spreads and shifts
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        best_bid_spread = (self.reference_price - Decimal(str(best_bid))) / self.reference_price
        best_ask_spread = (Decimal(str(best_ask)) - self.reference_price) / self.reference_price
        trend_shift = Decimal(self.price_multiplier) * Decimal(self.reference_price)
        inventory_shift = Decimal(self.inventory_multiplier) * Decimal(self.reference_price)
        sentiment_shift = Decimal(self.get_market_sentiment() * self.sentiment_impact_factor) * Decimal(self.reference_price)

        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend(["  Spreads:"])
        lines.extend([f"  Bid Spread (bps): {self.bid_spread * 10000:.4f} | Best Bid Spread (bps): {best_bid_spread * 10000:.4f}"])
        lines.extend([f"  Ask Spread (bps): {self.ask_spread * 10000:.4f} | Best Ask Spread (bps): {best_ask_spread * 10000:.4f}"])
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend(["  Price Shifts:"])
        lines.extend([f"  Max Shift (bps): {self.max_shift_spread * 10000:.4f}"])
        lines.extend([f"  Trend Scalar: {self.trend_scalar:.1f} | Trend Multiplier (bps): {self.price_multiplier * 10000:.4f} | Trend Shift: {trend_shift:.4f}"])
        lines.extend([f"  Target Inventory Ratio: {self.target_ratio:.4f} | Current Ratio: {self.current_ratio:.4f} | Inventory Delta: {self.inventory_delta:.4f}"])
        lines.extend([f"  Inventory Multiplier (bps): {self.inventory_multiplier * 10000:.4f} | Inventory Shift: {inventory_shift:.4f}"])
        lines.extend([f"  Market Sentiment: {self.market_sentiment:.2f} | Sentiment Shift: {sentiment_shift:.4f}"])
        lines.extend([f"  Original Price: {self.orig_price:.4f} | Reference Price: {self.reference_price:.4f}"])
        lines.extend(["\n----------------------------------------------------------------------\n"])
        
        # Volume Profile
        lines.extend(["  Volume Profile:"])
        lines.extend([f"  High Volume Levels: {len(self.high_volume_levels)} levels found"])
        if self.high_volume_levels:
            sorted_levels = sorted(self.high_volume_levels.items())
            for i, (level, volume) in enumerate(sorted_levels[:5]):  # Show top 5 levels
                lines.extend([f"    Level {i+1}: Price={level:.2f}, Volume={volume:.2f}"])
        lines.extend(["\n----------------------------------------------------------------------\n"])
        
        # Multi-timeframe Analysis
        lines.extend(["  Multi-timeframe Analysis:"])
        for tf in self.timeframes:
            if tf in self.candles_dict and not self.candles_dict[tf].candles_df.empty:
                df = self.candles_dict[tf].candles_df
                # Add RSI to dataframe
                if f"RSI_{self.candles_length}" not in df.columns:
                    df.ta.rsi(length=self.candles_length, append=True)
                
                if f"RSI_{self.candles_length}" in df.columns and not df.empty:
                    rsi_value = df[f"RSI_{self.candles_length}"].iloc[-1]
                    lines.extend([f"    {tf} RSI: {rsi_value:.2f} (Weight: {self.timeframe_weights[tf]:.2f})"])
        lines.extend(["\n----------------------------------------------------------------------\n"])
        
        # Main Candles
        candles_df = self.get_candles_with_features()
        lines.extend([f"  Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
        lines.extend(["    " + line for line in candles_df.tail(5).iloc[::-1].to_string(index=False).split("\n")])

        return "\n".join(lines)
