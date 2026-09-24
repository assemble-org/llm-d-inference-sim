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
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// streamHeadersProbe posts one streaming chat completion and reports when the
// response headers arrived, when the first data line arrived, and whether the
// stream ended with [DONE]. Header arrival is when client.Post returns.
func streamHeadersProbe(client *http.Client) (headersAt, firstDataAt time.Duration, status int, done bool) {
	body := fmt.Sprintf(`{"model":%q,"stream":true,"max_tokens":100,"messages":[{"role":"user","content":%q}]}`,
		common.TestModelName, testUserMessage)
	start := time.Now()
	resp, err := client.Post("http://localhost/v1/chat/completions", "application/json", strings.NewReader(body))
	Expect(err).NotTo(HaveOccurred())
	headersAt = time.Since(start)
	status = resp.StatusCode
	defer resp.Body.Close() //nolint:errcheck
	if status != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		Fail(fmt.Sprintf("unexpected status %d: %s", status, string(b)))
	}
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		if firstDataAt == 0 {
			firstDataAt = time.Since(start)
		}
		if strings.TrimPrefix(line, "data: ") == "[DONE]" {
			done = true
		}
	}
	return headersAt, firstDataAt, status, done
}

var _ = Describe("Streaming response headers timing", Ordered, func() {
	const ttft = 1200 * time.Millisecond

	It("holds the status and headers until the first token by default", func() {
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--time-to-first-token", ttft.String()})
		Expect(err).NotTo(HaveOccurred())

		headersAt, firstDataAt, status, done := streamHeadersProbe(client)
		Expect(status).To(Equal(http.StatusOK))
		Expect(done).To(BeTrue())
		Expect(headersAt).To(BeNumerically(">=", ttft-100*time.Millisecond))
		Expect(firstDataAt).To(BeNumerically(">=", ttft-100*time.Millisecond))
	})

	It("commits the status and headers at once with stream-headers-early", func() {
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--time-to-first-token", ttft.String(), "--stream-headers-early"})
		Expect(err).NotTo(HaveOccurred())

		headersAt, firstDataAt, status, done := streamHeadersProbe(client)
		Expect(status).To(Equal(http.StatusOK))
		Expect(done).To(BeTrue())
		Expect(headersAt).To(BeNumerically("<", ttft/2))
		Expect(firstDataAt).To(BeNumerically(">=", ttft-100*time.Millisecond))
	})

	It("delays the headers by time-to-headers, independently of the first token", func() {
		const toHeaders = 600 * time.Millisecond
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--time-to-first-token", ttft.String(), "--stream-headers-early",
			"--time-to-headers", toHeaders.String()})
		Expect(err).NotTo(HaveOccurred())

		headersAt, firstDataAt, status, done := streamHeadersProbe(client)
		Expect(status).To(Equal(http.StatusOK))
		Expect(done).To(BeTrue())
		Expect(headersAt).To(BeNumerically(">=", toHeaders-100*time.Millisecond))
		Expect(headersAt).To(BeNumerically("<", ttft-100*time.Millisecond))
		// time-to-headers does not delay the first token beyond time-to-first-token.
		Expect(firstDataAt).To(BeNumerically("<", ttft+toHeaders))
	})

	It("can be switched on at runtime through /admin/config", func() {
		client, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--time-to-first-token", ttft.String()})
		Expect(err).NotTo(HaveOccurred())

		resp, err := client.Post("http://localhost/admin/config", "application/json",
			strings.NewReader(`{"stream-headers-early": true, "time-to-headers": "0s"}`))
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		resp.Body.Close() //nolint:errcheck

		headersAt, _, status, done := streamHeadersProbe(client)
		Expect(status).To(Equal(http.StatusOK))
		Expect(done).To(BeTrue())
		Expect(headersAt).To(BeNumerically("<", ttft/2))
	})

	It("rejects a negative time-to-headers", func() {
		_, err := startServerWithArgs(context.TODO(), []string{"cmd", "--model", common.TestModelName,
			"--mode", common.ModeEcho, "--stream-headers-early", "--time-to-headers", "-1s"})
		Expect(err).To(HaveOccurred())
	})
})
