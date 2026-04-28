# HarmonyOS 修复经验沉淀

## 2026-03-12 代码评审修复：HUKS session 清理与伪 TLS pinning 收口

### 根因
- `SecurePrefs.ets` 在 `huks.initSession()` 成功后，如果 `finishSession()` 抛错，没有调用 `huks.abortSession()` 回收 handle，失败积累后会耗尽进程级 HUKS session 配额。
- `GatewaySession.ets` 保留了未使用的 `scope` 构造参数，容易让调用方误以为该依赖会参与会话行为。
- 当前 HarmonyOS `@ohos.net.webSocket` API 只提供 `caPath`、`clientCert`、`skipServerCertVerification` 等连接选项，并不提供服务器证书指纹回调或握手拦截能力；项目里保留的 `tlsFingerprintSha256` / `onTlsFingerprint` / trust prompt 流程因此完全不会生效。

### 修复动作
- 为 `encryptSecureValue()` 和 `decryptSecureValue()` 的 `finishSession()` 失败路径补上 `abortSession()` 回收。
- 删除 `GatewaySession` 中未使用的 `scope` 字段与构造参数。
- 移除未实现的 TLS fingerprint handler、endpoint 指纹字段、trust prompt 模型和对应 ViewModel 流程，避免暴露虚假的安全能力。

### 可复用规则
1. HUKS 会话式 API 只要拿到 handle，就必须为异常路径准备 `abortSession()`，否则偶发错误会逐步放大成全局加解密故障。
2. 安全相关接口如果没有真实接到平台能力，就不能保留“以后也许会用”的占位实现；这会直接误导调用方和审查者。
3. 当底层 SDK 不提供证书摘要回调或握手拦截时，不能声称已实现 certificate pinning；正确做法是先收口接口，等待平台能力具备后再重新设计。

## 2026-03-12 代码评审修复：重复断开通知与 HUKS 会话入参顺序

### 根因
- `GatewaySession.ets` 在 WebSocket `error` 事件里直接调用了 `onDisconnectedHandler`，但没有抑制随后必然到来的 `close` 事件，导致断开通知被触发两次。
- `connect()` 外层 `catch` 在关闭半建立态 socket 前也没有设置 `suppressNextCloseDisconnectNotice`，调用方和 `close` 回调会重复更新 UI。
- `SecurePrefs.ets` 把加解密数据放进了 `huks.initSession()` 的 `inData`，这偏离了 HUKS 约定的 `finishSession()`/`updateSession()` 数据通道。

### 修复动作
- 在 WebSocket `error` 处理器中先设置 `suppressNextCloseDisconnectNotice = true`，再主动通知断开。
- 在 `connect()` 外层异常清理 socket 前，按 `activeSocket !== null` 设置 close-notice 抑制，避免重复断开回调。
- 将 HUKS 加解密流程改为：`initSession()` 仅传算法参数和空 `inData`，实际明文/密文统一传给 `finishSession()`。

### 可复用规则
1. 对会显式触发断开回调的 socket 错误路径，必须同时考虑紧随其后的 `close` 事件，避免同一失败路径重复通知 UI。
2. 如果连接建立流程在异常处理中主动关闭 socket，且调用方已经消费该异常，就应抑制 `close` 事件的二次断开通知。
3. 使用 HUKS 会话式加解密时，`initSession()` 负责建立会话，业务负载数据应进入 `finishSession()` 或 `updateSession()`，不要依赖未文档化的 `initSession(inData)` 行为。

## 2026-03-12 代码评审修复：instanceId UUID 格式不合法

### 根因
- `SecurePrefs.ets` 的 `generateInstanceId()` 把 16 字节随机数拼成了 `8-4-4-16` 四段格式，只是“像 UUID”，并不符合 RFC 4122 的 `8-4-4-4-12` 五段结构。
- 该值会作为 `client.instanceId` 发送到 gateway，一旦链路上的服务或中间件开始校验 UUID 格式，连接就会被拒绝。

### 修复动作
- 将 `instanceId` 输出改为标准五段 UUID 格式。
- 同时设置 version/variant 位，确保生成结果符合 RFC 4122 v4，而不是仅修正连字符位置。

### 可复用规则
1. 任何对外发送、会被服务端或中间件消费的标识符，只要声称是 UUID，就必须生成标准 RFC 4122 格式，不能输出“近似 UUID”的自定义字符串。
2. 修复 UUID 问题时不要只看分段长度，还要同时校正 version 和 variant 位，避免留下“格式对了但语义仍不标准”的半修复状态。

## 2026-03-12 代码评审修复：locale 透传、敏感输入掩码、移除伪装 wrapper JAR

### 根因
- `GatewaySession.ets` 将所有非英文语言硬编码映射到 `CN` 区域，产生了 `fr-CN`、`ja-CN` 这类误导性的 locale 标签。
- `OnboardingPage.ets` 的 gateway token 输入框未设置 `InputType.Password`，敏感令牌会明文显示。
- 根目录 `hvigorw.jar` 只是文本占位文件，却伪装成 JAR，容易误导 IDE、CI 和后续维护者。

### 修复动作
- 将连接参数中的 `locale` 改为直接使用 `i18n.System.getSystemLocale()`，失败时回退到 `en-US`。
- 为 onboarding 中的 gateway token `TextInput` 增加 `.type(InputType.Password)`。
- 删除占位 `hvigorw.jar`，并在 `.gitignore` 中忽略，明确项目只使用 `hvigorw.js`。

### 可复用规则
1. locale 应优先透传系统提供的完整 BCP-47 标签，不要在业务层自行拼接语言-地区。
2. token、password、secret 等任何敏感文本输入，默认使用 `InputType.Password`，除非明确需要可见输入。
3. 不要提交“内容与扩展名不符”的占位文件；若项目不使用某种 wrapper 入口，应删除并在忽略规则中说明。

## 2026-03-12 代码评审修复：摘要实现与错误兜底本地化

### 根因
- `DeviceAuthStore.ets` 里内置了一套手写 SHA-256 实现来派生 `deviceId`，这类安全基础能力不应在业务代码中自行维护，既增加审查成本，也容易与平台能力脱节。
- `MainViewModel.ets` 仍有少量英文兜底错误直接写在 `throw new Error(...)` 和连接失败分支里，导致资源缺失以外的异常路径无法跟随系统语言切换。

### 修复动作
- 将 `DeviceAuthStore.ets` 的公钥摘要逻辑改为 HarmonyOS `cryptoFramework.createMd('SHA256')`，通过平台摘要 API 同步计算十六进制哈希。
- 删除不再需要的手写 `sha256Hex()`、`rotateRight()`、`wordToHex()` 逻辑，仅保留轻量 `bytesToHex()`。
- 将 `MainViewModel.ets` 中 “Unable to reach gateway.”、“Unable to load chat history.”、“Unable to load chat sessions.” 改为资源化兜底。
- 在 `base`、`en_US`、`zh_CN` 三套 `string.json` 中补充对应资源键。

### 可复用规则
1. 加密、摘要、签名等底层安全能力优先使用平台或成熟库提供的标准实现，避免在业务层维护手写密码学代码。
2. `throw new Error()` 内的 fallback 文案同样属于用户可见字符串，必须进入资源系统，不能只清理 UI 层硬编码。
3. 对“协议错误消息优先、资源兜底次之”的分支，统一使用 `response.error?.message || getLocalizedString(...)` 模式，兼顾服务端细节与本地化体验。

## 2026-03-12 Onboarding 最后一步连接成功后未即时刷新

### 根因
- `OnboardingPage.ets` 的最后一步直接读取 `viewModel.isConnected`、`viewModel.statusText`、`viewModel.serverName` 等字段，但页面本身没有像 `ConnectPage.ets` 那样订阅连接状态监听器。
- 连接流程完成后，`MainViewModel` 内部虽然已经更新了连接状态并广播 `notifyConnectionStateChanged()`，但引导页没有把这些变化同步到本地 `@State`，导致最后一步界面停留在旧的 `Connecting…` 状态。
- 当用户点击回退再前进时，页面重新执行构建逻辑，才间接读取到最新的连接结果。

### 修复动作
- 为 `OnboardingPage.ets` 增加连接状态监听注册与销毁。
- 新增本地 `@State`：`connectedState`、`statusState`、`serverNameState`、`remoteAddressState`。
- 新增 `syncConnectionState()`，在收到连接状态变化时同步本地状态，并在连接成功时主动清掉 `isConnecting` 与 `gatewayError`。
- 最后一步的状态文字、成功提示和按钮分支统一改为依赖这些本地同步状态。

### 可复用规则
1. 引导页、弹窗页、确认页这类一次性流程页面，如果依赖 ViewModel 异步状态，优先显式订阅对应 listener，而不要只依赖 `@ObjectLink` 的隐式刷新。
2. 对连接成功后需要立即切换按钮文案或分支 UI 的页面，必须保证“驱动 UI 的条件”来自本地 `@State` 或明确可观察的同步源。
3. 如果出现“切换页面后才显示正确状态”的现象，优先排查页面是否遗漏了 ViewModel 的状态监听，而不是先怀疑网络流程本身。

## 2026-03-12 剩余页面硬编码清理与语言响应式补齐

### 根因
- `ChatPage.ets`、`ConnectPage.ets`、`Index.ets` 仍残留少量硬编码文案或回退字符串，例如 `Not set`、`Loading device state…`、`main`、`See attached.`。
- 部分页面虽然已经使用了 `$r(...)`，但页面本身没有订阅 `localeVersion`，系统语言切换后当前页不一定会立刻重建。
- `MainViewModel.ets` 中的 `statusText`、`discoveryStatusText`、`micStatusText` 等状态文案长期保存为普通字符串，导致语言切换后这些状态值不会自动按新语言重算。

### 修复动作
- 为 `ChatPage.ets`、`ConnectPage.ets`、`Index.ets` 增加 `@StorageProp('localeVersion')`，并通过本地化 helper 读取资源。
- 补充 `common_not_set`、`index_loading_state`、`chat_see_attached`、`connect_saved_gateway`、`connect_discovery_ready`、`settings_ready` 等资源键。
- 将 `MainViewModel.ets` 中可枚举的状态文案统一改为“资源键 + 当前文本”双轨保存，通过 `setStatusTextFromResource()`、`setDiscoveryStatusFromResource()`、`setMicStatusFromResource()` 赋值。
- 在 `refreshLocalizedUiState()` 中重算资源型状态文案、重建主会话显示名和已保存网关名称，确保系统语言切换返回前台后立刻同步。

### 可复用规则
1. 只要页面需要跟随系统语言即时刷新，就不能只依赖资源文件本身，页面组件必须订阅一个显式的响应式语言版本信号。
2. ViewModel 中保存给 UI 展示的状态文案时，优先保存资源键或状态枚举，而不是只保存最终字符串；否则语言切换后无法无损重算。
3. 对于 `Main`、`Not set`、`Loading...`、`Saved Gateway` 这类很容易被忽略的小文案，也必须纳入资源系统，否则多语言体验会出现“局部漏翻译”。

## 2026-03-12 OnboardingPage 多语言改造

### 根因
- `OnboardingPage.ets` 中绝大多数用户可见文案都以硬编码英文形式直接写在页面与逻辑分支里。
- 这导致引导页既无法跟随资源语言切换，也无法和应用其余页面共享统一的多语言维护方式。
- 页面内还存在带占位符的动态文案，例如步骤进度、状态文本和连接目标，这类文案如果不统一抽到资源中，后续扩展语言会持续增加维护成本。

### 修复动作
- 为 `OnboardingPage.ets` 新增 `onboarding_*` 资源键，覆盖步骤标题、欢迎说明、网关设置、权限说明、检查页、按钮文案和错误提示。
- 同步更新 `base`、`en_US`、`zh_CN` 三套 `string.json`。
- 页面中新增 `localizedText()` 和若干轻量格式化方法，把步骤计数、状态文本、连接目标等动态文案改成基于资源模板的生成方式。
- 页面订阅 `localeVersion`，保证系统语言变化后返回应用时可立即重绘多语言文本。

### 可复用规则
1. 引导页、空状态页、错误页这类“看起来像临时页面”的用户界面，也必须走统一资源系统，不能因为页面独立就保留硬编码。
2. 含动态占位符的多语言文案应优先定义为资源模板，再通过小型 helper 替换 `{placeholder}`，避免在 UI 层拼接半英文半变量字符串。
3. ArkTS 严格模式下，避免把未声明类型的对象字面量直接传给通用格式化函数；优先拆成明确 helper，减少编译器类型限制问题。

## 2026-03-12 系统语言切换后页面未即时刷新

### 根因
- `SettingsPage.ets` 与 `PostOnboardingTabs.ets` 虽然已经通过资源系统读取多语言字符串，但页面本身没有订阅任何“语言已变化”的响应式状态。
- 用户从系统设置切换语言后返回应用，只触发了应用前后台生命周期，没有触发这些页面的局部状态变化，因此当前页不会立即重建。
- 页面切换或重启应用之所以能看到新语言，本质上是因为组件被重新创建后才重新读取了资源。

### 修复动作
- 在 `EntryAbility.ets` 中引入全局 `AppStorage` 键 `localeVersion`，并在 `onForeground()` 中递增，作为“系统语言可能已变化”的统一刷新信号。
- 在 `PostOnboardingTabs.ets` 与 `SettingsPage.ets` 中通过 `@StorageProp('localeVersion')` 订阅该信号。
- 在页面的 `localizedText()` 方法中读取 `localeVersion`，让所有依赖资源字符串的 UI 在语言切换返回应用时立即重建。
- 额外在 `MainViewModel.ets` 中补充 `refreshLocalizedUiState()`，用于在前台恢复时主动通知依赖连接/聊天状态的页面刷新。

### 可复用规则
1. HarmonyOS 多语言页面如果使用 `resourceManager.getStringByNameSync()` 手动取文案，必须同时绑定一个可触发重建的响应式状态源。
2. 系统语言切换这类全局配置变化，优先通过 `AppStorage` 下发全局版本号，再由页面使用 `@StorageProp` 订阅。
3. 对“切页后才刷新、重启后才生效”的多语言问题，优先排查组件是否缺少配置变化后的重建触发，而不是先怀疑资源文件本身。

## 2026-03-12 导航页裁剪与设置页多语言修复

### 根因
- `PostOnboardingTabs.ets` 仍保留未实现的 `VoicePage`、`ScreenPage` 入口，导致底部导航与当前产品范围不一致。
- 底部导航标签直接写死英文文案，未接入资源系统。
- `SettingsPage.ets` 大量标题、说明、占位符和状态兜底文案使用硬编码字符串，未走 `string.json` 多语言资源。

### 修复动作
- 移除 `VoicePage.ets` 与 `ScreenPage.ets`，并将底部导航裁剪为 `Connect`、`Chat`、`Settings` 三项。
- 为导航栏新增 `tab_*` 资源键，并在 `PostOnboardingTabs.ets` 中统一通过资源读取标签文案。
- 为 `SettingsPage.ets` 补充完整的 `settings_*` 资源键，覆盖页头、分组标题、开关说明、状态兜底与 About 信息。
- 同步更新 `base`、`en_US`、`zh_CN` 三套字符串资源，确保中英文一致。

### 可复用规则
1. 当某个能力页未实现或已下线时，必须同步移除：页面文件、导航入口、索引映射及相关状态联动逻辑。
2. HarmonyOS 页面中的用户可见文案禁止直接硬编码，必须优先写入 `entry/src/main/resources/*/element/string.json`。
3. 对需要在逻辑函数中返回文案的场景，可封装统一的本地化读取函数，并提供英文兜底值，避免资源缺失导致页面异常。
4. 修改底部导航项数量后，需同步调整 item 布局策略，优先使用等权重布局而非固定百分比宽度。

## 2026-03-12 代码评审修复：会话 key 与聊天列表稳定性

### 根因
- `GatewaySession.extractMainSessionKey()` 在响应缺少 `mainSessionKey` 时静默回退到固定值，可能把所有聊天流量错误路由到同一个伪造会话。
- `ChatPage.localizedThinkingLevel()` 直接返回英文硬编码，导致思考等级选项无法随语言切换。
- `ChatPage` 的消息列表使用数组下标作为 `ForEach` key，列表重建时无法稳定复用消息节点。

### 修复动作
- 将缺失 `mainSessionKey` 视为认证失败，直接抛错，阻止错误会话继续流转。
- 为聊天思考等级补充资源键，并通过资源读取显示文案。
- 为 `ChatMessage` 增加 `stableKey`，在消息解析与本地发送时写入稳定标识，并让 `ForEach` 使用该标识作为 key。

### 可复用规则
1. 协议层关键字段缺失时禁止静默兜底到固定常量，必须尽早抛错并走失败路径。
2. 任何会展示给用户的枚举文案都必须进入 `string.json`，不能留在 `switch` 分支里硬编码。
3. 列表渲染只要存在流式更新、局部插入或缓存组件，就必须优先使用业务稳定 key，避免使用数组下标。

## 2026-03-12 代码评审修复：认证就绪态与 Base64/大整数健壮性

### 根因
- `GatewaySession.ets` 在 WebSocket 握手成功后立即把 `connected` 置为 `true`，但认证仍在进行中；外部调用 `isConnected()` 或 `request()` 时可能误把“TCP 已连通”当成“gateway 已可用”。
- `awaitConnectNonce()` 默认只等 2 秒，移动网络下 `connect.challenge` 往返稍慢就会把健康连接误判成认证失败。
- `DeviceAuthStore.ets` 对超出目标长度的 bigint 直接截断高位，会把异常 key material 静默变成错误私钥。
- `DeviceAuthStore.ets` 与 `SecurePrefs.ets` 的 `decodeBase64()` 只识别标准 Base64 字符，遇到 URL-safe `-` / `_` 会悄悄跳过字符，最终触发隐蔽的数据损坏或误重建。

### 修复动作
- 将 `GatewaySession` 的“会话已就绪”定义收紧为 `connected && !connecting`，并在认证完成前拒绝除 `connect` 外的请求。
- 将 `awaitConnectNonce()` 默认超时提升到 8000 ms，降低移动网络下的误超时概率。
- 在 `bigIntToFixedBytesLE()` 中对超长输入直接抛错，避免静默截断密钥材料。
- 在两个 `decodeBase64()` 实现中先把 URL-safe Base64 规范化为标准字母表再解码。

### 可复用规则
1. 对需要先做 socket 握手再做协议认证的链路，公开的“已连接”状态必须表示“认证完成可发业务请求”，不能只表示底层传输已建立。
2. 移动端握手超时默认值要按高延迟网络设计，优先避免把慢链路误判成故障。
3. 涉及密钥材料或定长二进制数据时，长度溢出必须显式报错，不能靠截断兜底。
4. 只要系统里同时存在标准 Base64 和 Base64URL，底层解码 helper 就应统一先做字母表规范化，避免调用方各自踩坑。
