/**
 * ⑫ 2026-10-08 rag-ui-findings：会话消息区走 overlay 隐式滑条（照主仓成品：
 * StickToBottom + ScrollArea type="scroll"），并保留 role="log"（e2e 定位依赖）。
 */
import { afterEach, describe, expect, it } from "@rstest/core";
import { cleanup, render } from "@testing-library/react";

import { MessageList } from "@/components/workspace/messages";
import { I18nContext } from "@/core/i18n/context";
import { zhCN } from "@/core/i18n/locales/zh-CN";

function makeThread() {
  return {
    messages: [],
    isLoading: false,
    error: null,
    values: {},
    stop: () => undefined,
  };
}

describe("MessageList overlay scrollbars (⑫)", () => {
  afterEach(() => {
    cleanup();
  });

  it("wraps the transcript in the overlay viewport and keeps role=log", () => {
    const { container } = render(
      <I18nContext.Provider
        value={{ locale: "zh-CN", setLocale: () => undefined, t: zhCN }}
      >
        <MessageList
          className="size-full"
          threadId="thread-1"
          thread={makeThread() as never}
          hasMoreHistory={false}
          loadMoreHistory={() => undefined}
          isHistoryLoading={false}
        />
      </I18nContext.Provider>,
    );
    expect(container.querySelector('[role="log"]')).toBeTruthy();
    expect(
      container.querySelector('[data-slot="scroll-area-viewport"]'),
    ).toBeTruthy();
  });
});
