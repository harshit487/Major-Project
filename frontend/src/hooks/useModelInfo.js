import { useState, useEffect, useCallback } from 'react';
import { fetchHealthStatus, fetchModelInfo, extractApiErrorMessage } from '../services/api';

export function useModelInfo() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, infoData] = await Promise.all([
        fetchHealthStatus(),
        fetchModelInfo(),
      ]);
      setHealth(healthData);
      setModelInfo(infoData);
    } catch (err) {
      setError(extractApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { health, modelInfo, loading, error, refresh };
}
