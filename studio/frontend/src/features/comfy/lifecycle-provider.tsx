// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import { ComfyModelLifecycleContext } from "./lifecycle-context";

export function ComfyModelLifecycleProvider({
  ejectModel,
  children,
}: {
  ejectModel: () => Promise<boolean>;
  children: ReactNode;
}) {
  return (
    <ComfyModelLifecycleContext.Provider value={ejectModel}>
      {children}
    </ComfyModelLifecycleContext.Provider>
  );
}
