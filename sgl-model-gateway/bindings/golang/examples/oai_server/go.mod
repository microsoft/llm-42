module oai_server

go 1.25.0

replace github.com/sglang/sglang-go-grpc-sdk => ../..

require (
	github.com/sglang/sglang-go-grpc-sdk v0.0.0-00010101000000-000000000000
	github.com/valyala/fasthttp v1.52.0
	go.uber.org/zap v1.27.0
	gopkg.in/natefinch/lumberjack.v2 v2.2.1
)

require (
	github.com/andybalholm/brotli v1.1.0 // indirect
	github.com/klauspost/compress v1.17.9 // indirect
	github.com/stretchr/testify v1.10.0 // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	go.uber.org/multierr v1.10.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260526163538-3dc84a4a5aaa // indirect
	google.golang.org/grpc v1.83.2 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)
