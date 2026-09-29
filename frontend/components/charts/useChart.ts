"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { ChartData } from "./types";

type State = { data: ChartData | null; error: string | null; loading: boolean };

/**
 * Fetch one chart's numbers.
 *
 * Deliberately per-chart and on demand: the explainability page has sixteen
 * local explanations and loading all of them to show one would be silly.
 */
export function useChart(name: string | null): State {
  const [state, setState] = useState<State>({ data: null, error: null, loading: !!name });

  useEffect(() => {
    if (!name) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let live = true;
    setState({ data: null, error: null, loading: true });
    fetch(`${API_BASE}/api/charts/${name}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Not generated yet" : `API error ${r.status}`);
        return (await r.json()) as ChartData;
      })
      .then((d) => live && setState({ data: d, error: null, loading: false }))
      .catch((e) => live && setState({ data: null, error: e.message, loading: false }));
    // A stale response must not overwrite a newer one when the selector moves.
    return () => {
      live = false;
    };
  }, [name]);

  return state;
}
