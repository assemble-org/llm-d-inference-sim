/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tests

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// streamAbortProbe posts one streaming chat completion in echo mode and
// reports how many content chunks arrived, whether [DONE] arrived, and the
// read error the body ended with.
func streamAbortProbe(client *http.Client, prompt string) (contentChunks int, done bool, readErr error) {
	body := fmt.Sprintf(`{"model":%q,"stream":true,"max_tokens":100,"messages":[{"role":"user","content":%q}]}`,
		common.TestModelName, prompt)
	resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	Expect(resp.StatusCode).To(Equal(http.StatusOK))
	defer resp.Body.Close() //nolint:errcheck
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			continue
		}
		if strings.Contains(payload, `"content":"`) && !strings.Contains(payload, `"content":""`) {
			contentChunks++
		}
	}
	return contentChunks, done, scanner.Err()
}

var _ = Describe("Streaming abort after N tokens", Ordered, func() {
	const prompt = "one two three four five six seven eight nine ten"

	It("streams to completion with the knob off", func() {
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho})
		Expect(err).NotTo(HaveOccurred())
		chunks, done, readErr := streamAbortProbe(client, prompt)
		Expect(readErr).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(chunks).To(BeNumerically(">=", 10))
	})

	It("drops the connection after the configured number of token chunks", func() {
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--stream-abort-after-tokens", "3"})
		Expect(err).NotTo(HaveOccurred())
		chunks, done, readErr := streamAbortProbe(client, prompt)
		Expect(done).To(BeFalse(), "no [DONE] after an abort")
		Expect(chunks).To(BeNumerically("<=", 3))
		Expect(chunks).To(BeNumerically(">=", 1))
		// The chunked body has no terminating chunk, so the client's read
		// ends in an error, not a clean EOF.
		Expect(readErr).To(HaveOccurred())
	})
})
