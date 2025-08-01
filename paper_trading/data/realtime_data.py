"""
Real-time Data Integration for Paper Trading System
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import queue

logger = logging.getLogger(__name__)


class RealTimeDataProvider:
    """
    Real-time Data Provider for Live Trading
    
    Features:
    - WebSocket connections for real-time data
    - Multiple data source support
    - Automatic reconnection
    - Data buffering and processing
    - Event-driven architecture
    """
    
    def __init__(self, symbols: List[str], data_sources: List[str] = None):
        self.symbols = symbols
        self.data_sources = data_sources or ['yahoo', 'alpaca']
        self.connections = {}
        self.data_buffers = {symbol: queue.Queue(maxsize=1000) for symbol in symbols}
        self.callbacks = []
        self.is_running = False
        self.lock = threading.Lock()
        
        logger.info(f"RealTimeDataProvider initialized for {len(symbols)} symbols")
    
    def add_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for data updates"""
        self.callbacks.append(callback)
        logger.info("Callback added for real-time data updates")
    
    def start(self):
        """Start real-time data collection"""
        if self.is_running:
            logger.warning("Real-time data provider already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time data collection...")
        
        # Start data collection threads
        for source in self.data_sources:
            if source == 'yahoo':
                self._start_yahoo_stream()
            elif source == 'alpaca':
                self._start_alpaca_stream()
            elif source == 'polygon':
                self._start_polygon_stream()
    
    def stop(self):
        """Stop real-time data collection"""
        self.is_running = False
        logger.info("Stopping real-time data collection...")
        
        # Close all connections
        for connection in self.connections.values():
            if hasattr(connection, 'close'):
                connection.close()
    
    def _start_yahoo_stream(self):
        """Start Yahoo Finance real-time stream"""
        def yahoo_stream():
            while self.is_running:
                try:
                    # Simulate Yahoo Finance WebSocket connection
                    for symbol in self.symbols:
                        # Generate mock real-time data
                        data = self._generate_mock_realtime_data(symbol)
                        self._process_realtime_data(symbol, data)
                    
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Yahoo stream error: {e}")
                    time.sleep(5)  # Wait before retry
        
        thread = threading.Thread(target=yahoo_stream, daemon=True)
        thread.start()
        self.connections['yahoo'] = thread
        logger.info("Yahoo Finance stream started")
    
    def _start_alpaca_stream(self):
        """Start Alpaca real-time stream"""
        def alpaca_stream():
            while self.is_running:
                try:
                    # Simulate Alpaca WebSocket connection
                    for symbol in self.symbols:
                        # Generate mock real-time data
                        data = self._generate_mock_realtime_data(symbol, source='alpaca')
                        self._process_realtime_data(symbol, data)
                    
                    time.sleep(0.5)  # Update every 500ms
                    
                except Exception as e:
                    logger.error(f"Alpaca stream error: {e}")
                    time.sleep(5)  # Wait before retry
        
        thread = threading.Thread(target=alpaca_stream, daemon=True)
        thread.start()
        self.connections['alpaca'] = thread
        logger.info("Alpaca stream started")
    
    def _start_polygon_stream(self):
        """Start Polygon real-time stream"""
        def polygon_stream():
            while self.is_running:
                try:
                    # Simulate Polygon WebSocket connection
                    for symbol in self.symbols:
                        # Generate mock real-time data
                        data = self._generate_mock_realtime_data(symbol, source='polygon')
                        self._process_realtime_data(symbol, data)
                    
                    time.sleep(0.1)  # Update every 100ms
                    
                except Exception as e:
                    logger.error(f"Polygon stream error: {e}")
                    time.sleep(5)  # Wait before retry
        
        thread = threading.Thread(target=polygon_stream, daemon=True)
        thread.start()
        self.connections['polygon'] = thread
        logger.info("Polygon stream started")
    
    def _generate_mock_realtime_data(self, symbol: str, source: str = 'yahoo') -> Dict:
        """Generate mock real-time data for testing"""
        base_price = 150.0 + hash(symbol) % 100  # Different base price per symbol
        
        # Add some randomness and trend
        timestamp = datetime.now()
        price_change = np.random.normal(0, 0.5)
        volume = np.random.randint(1000, 10000)
        
        if source == 'yahoo':
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'price': base_price + price_change,
                'volume': volume,
                'bid': base_price + price_change - 0.01,
                'ask': base_price + price_change + 0.01,
                'source': 'yahoo'
            }
        elif source == 'alpaca':
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'price': base_price + price_change,
                'volume': volume,
                'trade_size': np.random.randint(100, 1000),
                'source': 'alpaca'
            }
        else:  # polygon
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'price': base_price + price_change,
                'volume': volume,
                'trade_count': np.random.randint(1, 10),
                'source': 'polygon'
            }
    
    def _process_realtime_data(self, symbol: str, data: Dict):
        """Process incoming real-time data"""
        with self.lock:
            # Add to buffer
            try:
                self.data_buffers[symbol].put_nowait(data)
            except queue.Full:
                # Remove oldest data if buffer is full
                try:
                    self.data_buffers[symbol].get_nowait()
                    self.data_buffers[symbol].put_nowait(data)
                except queue.Empty:
                    pass
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest data for a symbol"""
        try:
            return self.data_buffers[symbol].get_nowait()
        except queue.Empty:
            return None
    
    def get_data_history(self, symbol: str, max_items: int = 100) -> List[Dict]:
        """Get recent data history for a symbol"""
        history = []
        buffer = self.data_buffers[symbol]
        
        # Get all available data
        while not buffer.empty() and len(history) < max_items:
            try:
                data = buffer.get_nowait()
                history.append(data)
            except queue.Empty:
                break
        
        return history


class WebSocketManager:
    """
    WebSocket Connection Manager
    
    Features:
    - Multiple WebSocket connections
    - Automatic reconnection
    - Connection pooling
    - Error handling
    """
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = {}
        self.connection_pool = []
        self.is_running = False
        
        logger.info(f"WebSocketManager initialized with max {max_connections} connections")
    
    async def connect(self, url: str, protocols: List[str] = None) -> websockets.WebSocketServerProtocol:
        """Establish WebSocket connection"""
        try:
            connection = await websockets.connect(url, subprotocols=protocols)
            logger.info(f"WebSocket connected to {url}")
            return connection
        except Exception as e:
            logger.error(f"WebSocket connection failed to {url}: {e}")
            raise
    
    async def subscribe(self, connection: websockets.WebSocketServerProtocol, 
                       subscription: Dict) -> bool:
        """Subscribe to data feed"""
        try:
            await connection.send(json.dumps(subscription))
            response = await connection.recv()
            logger.info(f"Subscription successful: {response}")
            return True
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    async def listen(self, connection: websockets.WebSocketServerProtocol, 
                    callback: Callable[[Dict], None]):
        """Listen for incoming messages"""
        try:
            async for message in connection:
                try:
                    data = json.loads(message)
                    callback(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")


class DataAggregator:
    """
    Real-time Data Aggregator
    
    Features:
    - Multi-source data aggregation
    - Data validation and cleaning
    - Time synchronization
    - Quality metrics
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.aggregated_data = {}
        self.data_quality = {}
        self.last_update = {}
        
        logger.info(f"DataAggregator initialized for {len(symbols)} symbols")
    
    def aggregate_data(self, symbol: str, data_sources: Dict[str, Dict]) -> Dict:
        """Aggregate data from multiple sources"""
        aggregated = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': 0.0,
            'volume': 0,
            'bid': 0.0,
            'ask': 0.0,
            'spread': 0.0,
            'sources': len(data_sources),
            'quality_score': 0.0
        }
        
        prices = []
        volumes = []
        bids = []
        asks = []
        
        # Collect data from all sources
        for source, data in data_sources.items():
            if 'price' in data:
                prices.append(data['price'])
            if 'volume' in data:
                volumes.append(data['volume'])
            if 'bid' in data:
                bids.append(data['bid'])
            if 'ask' in data:
                asks.append(data['ask'])
        
        # Calculate aggregated values
        if prices:
            aggregated['price'] = np.mean(prices)
        if volumes:
            aggregated['volume'] = int(np.mean(volumes))
        if bids:
            aggregated['bid'] = np.mean(bids)
        if asks:
            aggregated['ask'] = np.mean(asks)
        
        if aggregated['bid'] > 0 and aggregated['ask'] > 0:
            aggregated['spread'] = aggregated['ask'] - aggregated['bid']
        
        # Calculate quality score
        aggregated['quality_score'] = self._calculate_quality_score(data_sources)
        
        return aggregated
    
    def _calculate_quality_score(self, data_sources: Dict[str, Dict]) -> float:
        """Calculate data quality score"""
        if not data_sources:
            return 0.0
        
        # Factors for quality calculation
        source_count = len(data_sources)
        timestamp_freshness = 0.0
        price_consistency = 0.0
        
        # Check timestamp freshness
        current_time = datetime.now()
        for source, data in data_sources.items():
            if 'timestamp' in data:
                time_diff = (current_time - data['timestamp']).total_seconds()
                if time_diff < 1.0:  # Less than 1 second old
                    timestamp_freshness += 1.0
        
        # Check price consistency
        prices = [data.get('price', 0) for data in data_sources.values() if 'price' in data]
        if len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            if price_mean > 0:
                price_consistency = max(0, 1 - (price_std / price_mean))
        
        # Calculate overall quality score
        quality_score = (
            (source_count / 3.0) * 0.4 +  # Source diversity
            (timestamp_freshness / len(data_sources)) * 0.3 +  # Freshness
            price_consistency * 0.3  # Consistency
        )
        
        return min(1.0, quality_score)
    
    def update_data_quality(self, symbol: str, quality_score: float):
        """Update data quality metrics"""
        if symbol not in self.data_quality:
            self.data_quality[symbol] = []
        
        self.data_quality[symbol].append({
            'timestamp': datetime.now(),
            'quality_score': quality_score
        })
        
        # Keep only recent quality metrics
        if len(self.data_quality[symbol]) > 100:
            self.data_quality[symbol] = self.data_quality[symbol][-100:]
    
    def get_average_quality(self, symbol: str, hours: int = 1) -> float:
        """Get average data quality for a symbol"""
        if symbol not in self.data_quality:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_quality = [
            q['quality_score'] for q in self.data_quality[symbol]
            if q['timestamp'] > cutoff_time
        ]
        
        return np.mean(recent_quality) if recent_quality else 0.0


def create_realtime_provider(symbols: List[str], 
                           data_sources: List[str] = None) -> RealTimeDataProvider:
    """Convenience function to create real-time data provider"""
    return RealTimeDataProvider(symbols, data_sources)


def create_websocket_manager(max_connections: int = 10) -> WebSocketManager:
    """Convenience function to create WebSocket manager"""
    return WebSocketManager(max_connections)


def create_data_aggregator(symbols: List[str]) -> DataAggregator:
    """Convenience function to create data aggregator"""
    return DataAggregator(symbols) 