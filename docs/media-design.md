# Media: image recognition and generation (design proposal)

> **Status: proposal — not implemented.** This records the design direction for
> adding image input (recognition) and image generation to sefia, the
> alternatives considered, and why they were rejected. Names are provisional.

## Summary

One concept is added to the core: a **`Media` value type** — a *reference* to an
image (or, later, other media), never the bytes themselves.

- **Recognition** (the model sees an image) is the only part that needs core
  changes: `Media` values appearing in call arguments or tool results are
  rendered as image content parts in the messages sent to the model, and the
  provider adapter resolves the reference into the provider's wire format.
- **Generation is always a tool.** A `sefios` toolkit calls an image API and
  returns a `Media` reference. Native image-*output* models (e.g. Gemini's
  image models) are wrapped as tools too, not integrated into the decision
  loop.
- **History stays light.** Only references enter the step history; bytes live
  in a durable, content-addressed media store, which keeps glyff snapshots
  small and replay stable.

The decision schema stays what it is today: a constraint on the model's
*decision*. A decision about an image can only ever be a reference to one, so
`Media` may appear in schemas as plain structured data (a URI the model copies
from a tool result), but no machinery ever routes image payloads through the
schema.

## Design principle: schemas constrain decisions, not payloads

sefia's unified structured-output schema exists to give the model's decision a
verifiable shape ([how-it-works.md](./how-it-works.md)). Pixels have no
verifiable shape — validation can say nothing about them. The only judgment a
model can express about an image in a decision is *which* image: a reference.

This split resolves every design question below:

- References (`Media{uri, mime_type}`) are ordinary data. They may appear in
  arguments, tool results, and return types with zero special machinery — the
  model reads and writes URIs as strings inside JSON, exactly like any other
  field.
- Payloads (bytes) never enter a decision, the history, or a schema. They move
  through two channels only: *into* the model as content parts built at the
  client boundary, and *out of* tools as stored blobs behind a fresh reference.

## The `Media` type

A small dependency-free value type in `sefia` core:

```python
@dataclass(frozen=True)
class Media:
    uri: str                     # RFC 3986; data: URIs (RFC 2397) allowed for inline bytes
    mime_type: str | None = None # IANA MIME string, e.g. "image/png"; sniffed when omitted
    name: str | None = field(default=None, compare=False)  # descriptive, not identity
```

Every field is a stable, standards-backed format — URIs and MIME types are the
one representation that will not churn under provider changes, and the shape
maps mechanically onto Anthropic/OpenAI content parts and MCP resource types.

- `uri` covers remote (`https://`), local (`file://`), inline (`data:`), and
  store-backed (scheme TBD, e.g. `media://<hash>`) images in one field.
- Both fields are deliberately **plain `str`**, not wrapper types. Python has
  no standard-library URI type, so "a URI type" means adopting some parser's
  opinions — and the candidates (WHATWG-based, e.g. pydantic's `Url` types)
  *normalize*: schemes and authority are case-folded, which would silently
  mangle a case-sensitive content hash sitting in the authority position of a
  `media://` URI, and `data:` (huge inline payloads, no authority),
  `file://` (empty authority), and custom schemes all live at the poorly
  specified edge of WHATWG URL semantics. The persisted and wire form is the
  string either way; RFC 3986 itself is the stable standard, a wrapper type is
  a library opinion. Validation happens at the edges — the adapter or media
  store fails on a URI it cannot resolve; construction does at most a cheap
  has-a-scheme sanity check and never normalizes. `mime_type` is likewise a
  plain IANA MIME string: an open set with optional parameters, not an enum.
- The field is named `mime_type`, not `media_type`, even though IANA's own
  term is "media type" (RFC 6838): on a class named `Media`, `media_type`
  stutters and invites the misreading "the kind of this `Media`" (image vs
  audio), while `mime_type` unambiguously means the `"image/png"` string and
  matches the implementation-side vocabulary (MCP's `mimeType`, Python's
  `mimetypes`). The value itself is the IANA string either way.
- `name` is optional, descriptive metadata — a human/model-readable handle
  ("login-page-screenshot"), needed because the design's mainstream URIs
  (`data:`, store-backed `media://<hash>`) have no basename, and future
  document support needs a filename/title for provider file blocks anyway. It
  is excluded from equality (`compare=False`): two `Media` referencing the
  same bytes are the same media regardless of what someone called them, so
  naming can never confuse content-addressed deduplication.
- Explicitly a *value*, not an annotation: whether something is an image is a
  property of the value, so it survives being nested in lists and returned
  from tools, and expansion into context (which costs real tokens) is always
  an explicit opt-in — a bare string that happens to end in `.png` is never
  auto-expanded. Extension/scheme sniffing as the *trigger* was rejected for
  exactly that implicitness; sniffing only ever fills in a missing
  `mime_type` for a value that is already a `Media`.

## Recognition: `Media` is a message-content primitive

The model sees an image if and only if image content reaches the messages sent
to the `LLMClient`. Whether the `Media` arrived as a call argument or a tool
result is irrelevant — rendering is the whole feature.

The client-facing content model treats `Media` as a primitive alongside `str`:
`Message.content` is formalized from today's open-ended `str | list[Any]` to
**`str | list[ContentPart]`**, where **`ContentPart = str | Media`**. This is
sefia's counterpart of the content-part unions every provider API has
(OpenAI input items, Anthropic content blocks, Gemini `Part`, Bedrock
`ContentBlock`): `str` plays the text variant, `Media` the non-text one, and
in Python the type itself is the discriminator — no `{"type": ...}` tag
needed. The strategy places `Media` values into message content *as-is*;
translating them into a provider wire shape is the adapter's job, exactly
like the rest of `Message`/`LLMResponse`. The alternative — the strategy
pre-converting to some "neutral" content-part dict — was rejected: any such
dict is a provider wire format in disguise, leaking into the core, and it
would rob the adapter of the choice between passing a URL through and
inlining base64, which is provider knowledge.

Changes:

| Layer | Change |
| --- | --- |
| `llm/_strategy.py` (`_build_messages`) | Place `Media` from prompt arguments and tool results into message content parts, untranslated. |
| `llm/_messages.py` | Add `ContentPart = str \| Media`; tighten `Message.content` to `str \| list[ContentPart]`. |
| `sefia_litellm` | Walk content parts and translate each `Media` into the provider's format: pass URLs through where the provider accepts them; read and base64 `file://` / store-backed URIs; forward `data:` URIs. A blind `msg.to_dict()` no longer suffices — a serialized `Media` dataclass is not a valid content part on any wire. |

### Which MIME types render

The `Media` type never restricts `mime_type` (an open set, by decision above).
What is bounded is the *rendering path*: wrapping a value in `Media` is the
opt-in to expand it into context, and the MIME family selects how —

| MIME family | Rendering |
| --- | --- |
| `image/*` | image content parts (this proposal) |
| `application/pdf` | provider document blocks — native support exists, same boundary-translation shape as images (follow-up) |
| `text/*` (csv, markdown, plain, …) | inlined as text parts — well-defined and provider-independent (follow-up) |
| anything else (docx, xlsx, …) | no *universal* rendering — fail loudly by default; adapter-extensible (below) |

The criterion is not the file-format category but whether **boundary
translation alone** turns the reference into something the model perceives.
That makes the last row adapter-relative, not absolute: office formats have no
universal rendering, but where a provider accepts them natively (Bedrock's
Converse API takes docx/csv/xlsx as document blocks), its adapter *is* doing
pure boundary translation and may render them. The default elsewhere stays
fail-loud, and a run that must stay provider-portable converts explicitly
instead — a conversion with real choices in it (extract text and lose layout,
or convert to PDF) is processing, a tool's job
(`convert_document(media) -> Media`, returning a renderable PDF or
`text/plain`). The opt-in principle also settles the text case: a `file://`
reference you do *not* want read stays a plain `str` argument; wrapping it in
`Media` *is* the request to read it, so `text/*` inlining does not reintroduce
implicit expansion.

Two contract points follow:

- **A client that cannot render media must fail loudly.** The `LLMClient`
  contract states that an implementation receiving a `Media` part it does not
  support raises, rather than silently serializing it into nonsense.
- **The tool-role wrinkle lives in the adapter.** Most providers reject image
  parts inside `role="tool"` messages; the standard workaround (a textual
  reference in the tool message, the image in an immediately following `user`
  message) is a provider constraint, so the adapter applies it. The strategy
  just puts the `Media` where it logically belongs — in the tool result.

Sub-agent composition falls out for free: an `@infer` method that returns
`Media` (a reference) hands its parent agent something the parent's next step
can *see*, because the reference lands in history and rendering applies.

## Generation: always a tool

A `sefios` toolkit (extra-gated, e.g. `sefios[images]`) exposes generation as
an ordinary tool: `generate_image(prompt, ...) -> Media`. The implementation
calls an image API through a thin client seam (an `ImageGenClient` ABC with a
LiteLLM-backed implementation in `sefia_litellm`, mirroring how `LLMClient`
keeps provider specifics in the adapter), stores the returned bytes in the
media store, and returns the reference.

This framing carries further than it looks:

- **Engraving comes for free.** Tool batches are engraved, so an expensive
  generation never re-runs on resume — the stored `Media` replays.
- **Native image-output models are tools too.** A model that emits images in
  its response (Gemini image models and successors) is wrapped exactly like a
  DALL-E-style endpoint: from the tool boundary both are `prompt (+ reference
  images) → Media`. The decision loop's model stays a text/JSON model.
- **Iterate-by-looking already works at step granularity.** The loop *is* a
  coarse-grained interleave: generate (tool) → the `Media` enters history →
  the next step renders it → the model critiques and regenerates. Where a
  provider supports native interleaved generate-inspect-refine within one
  response, the wrapped tool can use it *internally* and return only the final
  image — a tool-implementation detail the core never learns about.

## Persistence: references in history, bytes in a store

Step history is persisted after every step into glyff metadata
([how-it-works.md](./how-it-works.md)); base64 payloads there would bloat
every snapshot. So:

- Only `Media` references enter `StepHistory` and engraved outputs.
- Bytes live in a **media store**: durable (engraved steps reference these
  URIs, so replay requires them to keep resolving) and content-addressed
  (hash-keyed URIs make replay stable and deduplicate repeated images),
  aligned with glyff's own content-addressing.
- Lifecycle (GC when a session is deleted) is the media store's concern, a
  deliberate new responsibility this design accepts.

Two follow-ons ride on the same structure:

- **Compaction**: images are the heaviest context items, so the
  `HistoryCompactor` grows a rule for demoting old images to their reference
  (or a cached caption) while recent steps keep the rendered image.
- **Cost accounting**: image input tokens and per-image generation prices are
  handled where cost already lives (`sefios/handlers/_cost.py` and the
  adapters).

## Alternatives considered and rejected

| Alternative | Why rejected |
| --- | --- |
| **Caption tool** — a tool calls a separate vision model and returns a text description; the main model never sees pixels. | Lossy: anything the caption omits is unrecoverable, and "look again" is impossible. Fine as a stopgap; not the design. (The flaw is information loss, not the extra tool call.) |
| **Sniff strings by extension/scheme** to decide what is an image. | Implicit and false-positive-prone; expansion must be an explicit opt-in (see above). |
| **A kind-discriminated content-part union in core** (`ImageContent \| DocumentContent \| AudioContent \| …` with a separate `url \| file_id \| inline` source axis), mirroring provider ContentBlock/Part shapes. | The kind vocabulary looks shared across providers but the membership rules disagree — CSV is a `document` on Bedrock, unsupported on Anthropic, plain text on Gemini; PDF is `document` / `input_file` / `fileData` depending on provider — so *kind* is a provider treatment category, derived by the adapter's mime→kind table, not a stable fact about the content (same reasoning as the unified decision schema vs native tool-calling, [tradeoffs.md](./tradeoffs.md)). The source axis adds nothing over URIs (`https:`/`data:` already encode url-vs-inline), and `file_id` is provider+account+TTL-scoped state that would poison durable, provider-portable history — provider upload handles are an adapter-level cache keyed by content hash. The message-level union sefia does need already exists as `ContentPart = str \| Media`. |
| **Name the type `Resource`** (MCP's term for the same `{uri, name, mimeType}` shape). | Names the mechanism (URI-addressability) instead of the semantic (perceivable content rendered into context, at token cost). A `file://` reference to a config file is a resource but must not be image-rendered, so the "every value of this type gets rendered" contract would be wrong for the name. Also collides with REST/cloud/RAII usage and with MCP's Resource, which is a different lifecycle concept (a listable, subscribable catalog entry). `Media` keeps the terminology straight next to `mime_type` — MIME's own registry is "media types". |
| **Native image output routed through the decision schema** — mark `Media` fields in the JSON Schema, enable the image modality, have the model reference emitted images via sentinel URIs (`attachment://n`), and let the adapter rewrite sentinels to stored URIs. | Workable but over-engineered: it turns the decision schema into a smuggling channel for binary payloads, adds vendor markers, a correlation protocol, and (on some providers) costs strict JSON mode — all to save one tool-call hop over wrapping the same model as a tool. Contradicts the schema-constrains-decisions principle. |
| **`LLMResponse.images` / images as first-class model output** in the loop. | Same conflict with the unified JSON decision, without even the schema to anchor it. |
| **Interleaved generate-inspect-refine in the core loop.** | Real (research systems and current provider APIs support interleaved output), but the step loop already provides it coarsely, and the fine-grained version is available *inside* a wrapped tool. Not worth a core mechanism. |

## Phasing

1. **Generation first (no core changes):** `ImageGenClient` seam +
   LiteLLM-backed implementation + `sefios[images]` generation toolkit + the
   media store. Usable immediately; the model handles `Media` as opaque
   reference data.
2. **Recognition (the core change):** `Media` type in `sefia`,
   `_build_messages` rendering, adapter-side URI resolution, the tool-role
   workaround.
3. **Hygiene:** image-aware compaction rule; image cost accounting.

Images are the scope of this proposal. PDF and `text/*` rendering are named
follow-ups (see the rendering matrix above); audio and further formats extend
the same shape without a new concept.

## See also

- [how-it-works.md](./how-it-works.md) — the loop, the unified schema, history
  persistence this design plugs into.
- [tradeoffs.md](./tradeoffs.md) — why sefia avoids provider-native tool
  calling; the same reasoning drives keeping provider media formats in the
  adapter.
- [architecture.md](./architecture.md) — the layering rules this design
  preserves (core knows interfaces; provider specifics live in adapters;
  batteries in `sefios`).
