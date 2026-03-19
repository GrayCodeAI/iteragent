package iteragent

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type BedrockConfig struct {
	Region      string
	Model       string
	AccessKey   string
	SecretKey   string
	MaxTokens   int
	Temperature float32
}

type BedrockProvider struct {
	config BedrockConfig
	client *http.Client
}

func NewBedrock(config BedrockConfig) *BedrockProvider {
	return &BedrockProvider{
		config: config,
		client: &http.Client{},
	}
}

func (p *BedrockProvider) Name() string {
	return "bedrock"
}

// TODO: Add ThinkingLevel support for Bedrock when provider supports it.
func (p *BedrockProvider) Complete(ctx context.Context, messages []Message, opts ...CompletionOptions) (string, error) {
	url := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/converse",
		p.config.Region, p.config.Model)

	systemPrompts := []map[string]string{}
	var convMessages []map[string]interface{}

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompts = append(systemPrompts, map[string]string{"text": msg.Content})
		} else {
			role := msg.Role
			if role == "assistant" {
				role = "assistant"
			}
			convMessages = append(convMessages, map[string]interface{}{
				"role": role,
				"content": []map[string]string{
					{"text": msg.Content},
				},
			})
		}
	}

	body := map[string]interface{}{
		"messages": convMessages,
	}
	if len(systemPrompts) > 0 {
		body["system"] = systemPrompts
	}
	if p.config.MaxTokens > 0 {
		body["maxTokens"] = p.config.MaxTokens
	}
	if p.config.Temperature > 0 {
		body["temperature"] = p.config.Temperature
	}

	jsonBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", err
	}

	p.signRequest(req, string(jsonBody))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Bedrock error (%d): %s", resp.StatusCode, string(respBody))
	}

	var response struct {
		Output struct {
			Message struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"message"`
		} `json:"output"`
	}

	if err := json.Unmarshal(respBody, &response); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	if len(response.Output.Message.Content) == 0 {
		return "", fmt.Errorf("no response content")
	}

	return response.Output.Message.Content[0].Text, nil
}

// CompleteStream implements TokenStreamer for Bedrock. Bedrock's streaming API uses
// HTTP/2 binary event framing which requires the AWS SDK to decode correctly.
// As a pragmatic fallback, this calls Complete and delivers the full response as
// a single token so the agent loop still benefits from retry logic.
func (p *BedrockProvider) CompleteStream(ctx context.Context, messages []Message, opt CompletionOptions, onToken func(string)) (string, error) {
	result, err := p.Complete(ctx, messages, opt)
	if err != nil {
		return "", err
	}
	if onToken != nil {
		onToken(result)
	}
	return result, nil
}

func (p *BedrockProvider) Stream(ctx context.Context, config StreamConfig, messages []Message, onEvent func(StreamEvent)) (Message, error) {
	url := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/converse",
		p.config.Region, p.config.Model)

	systemPrompts := []map[string]string{}
	var convMessages []map[string]interface{}

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompts = append(systemPrompts, map[string]string{"text": msg.Content})
		} else {
			convMessages = append(convMessages, map[string]interface{}{
				"role": msg.Role,
				"content": []map[string]string{
					{"text": msg.Content},
				},
			})
		}
	}

	body := map[string]interface{}{
		"messages": convMessages,
		"stream":   true,
	}
	if len(systemPrompts) > 0 {
		body["system"] = systemPrompts
	}
	if config.MaxTokens > 0 {
		body["maxTokens"] = config.MaxTokens
	}
	if config.Temperature > 0 {
		body["temperature"] = config.Temperature
	}

	jsonBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return Message{}, err
	}

	p.signRequest(req, string(jsonBody))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

	var content strings.Builder

	decoder := NewSSEDecoder(resp.Body)
	for {
		event, err := decoder.Decode()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}

		if event.Type == "content" {
			content.WriteString(event.Content)
			onEvent(event)
		}
	}

	return Message{
		Role:    "assistant",
		Content: content.String(),
	}, nil
}

func (p *BedrockProvider) signRequest(req *http.Request, payload string) {
	now := time.Now().UTC()
	date := now.Format("20060102T150405Z")
	amzDate := os.Getenv("AWS_AMZ_DATE")
	if amzDate != "" {
		date = amzDate
	}

	region := p.config.Region
	service := "bedrock"

	req.Header.Set("X-Amz-Date", date)
	req.Header.Set("X-Amz-Target", "AmazonBedrockRuntime.Converse")

	host := req.Host
	if host == "" {
		host = strings.Split(strings.Split(req.URL.Host, ":")[0], ".")[0]
	}

	hashedPayload := fmt.Sprintf("%x", sha256.Sum256([]byte(payload)))

	headers := []string{
		"content-type:application/json",
		fmt.Sprintf("host:%s", host),
		fmt.Sprintf("x-amz-date:%s", date),
		fmt.Sprintf("x-amz-target:AmazonBedrockRuntime.Converse"),
	}
	signedHeaders := "content-type;host;x-amz-date;x-amz-target"

	canonicalRequest := strings.Join([]string{
		"POST",
		"/model/" + p.config.Model + "/converse",
		"",
		strings.Join(headers, "\n"),
		signedHeaders,
		hashedPayload,
	}, "\n")

	hashedCanonical := fmt.Sprintf("%x", sha256.Sum256([]byte(canonicalRequest)))

	algorithm := "AWS4-HMAC-SHA256"
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", date[:8], region, service)

	stringToSign := strings.Join([]string{
		algorithm,
		date,
		credentialScope,
		hashedCanonical,
	}, "\n")

	kDate := hmacSHA256([]byte("AWS4"+p.config.SecretKey), date[:8])
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	kSigning := hmacSHA256(kService, "aws4_request")

	signature := fmt.Sprintf("%x", hmacSHA256(kSigning, stringToSign))

	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, p.config.AccessKey, credentialScope, signedHeaders, signature)

	req.Header.Set("Authorization", authHeader)
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

func init() {
	registry := NewProviderRegistry()
	registry.Register(ProtocolBedrock, &BedrockProvider{})
}
