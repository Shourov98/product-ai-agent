# Multi-Agent Product Generation System

## Overview

This system uses a deterministic multi-agent pipeline to transform
product image and title into marketplace-specific listing data.

Agents operate sequentially under a central orchestrator.

---

## Agent Architecture

### 1. Vision Agent

**Responsibility:**

- Background removal
- Product type detection
- Attribute extraction (color, material, style)

**Input:**

- Image file

**Output:**

- Structured vision metadata

---

### 2. Core Product Agent

**Responsibility:**

- Combine title + vision output
- Create normalized product schema
- Define features and category

**Input:**

- Title
- Vision metadata

**Output:**

- Standardized product object

---

### 3. Amazon Listing Agent

**Responsibility:**

- Generate SEO title
- Bullet points
- Description
- Backend search terms
- Structured attributes

---

### 4. TikTok Listing Agent

**Responsibility:**

- Generate short viral title
- Social description
- Hashtags

---

### 5. eBay Listing Agent

**Responsibility:**

- 80-character optimized title
- Item specifics
- Condition mapping

---

## Orchestration Model

Agents are executed sequentially:

Vision → Core → Marketplace Agents

The orchestrator guarantees:

- Structured JSON output
- Error isolation
- Validation before persistence

---

## Design Principles

- Deterministic workflow (no autonomous reasoning)
- Strict JSON schema enforcement
- Stateless agents
- Easy testability
- Extensible marketplace modules
