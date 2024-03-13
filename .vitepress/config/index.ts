import { defineConfig } from "vitepress"
import shared from "./shared"
import en_us, { sidebarGuide as EN_side } from "./en-us"
import zh_cn, { sidebarGuide as ZH_side } from "./zh-cn"
import handleRewrites from "../utils/rewrites"

const en = { lang: "en-us", list: EN_side() }
const zh = { lang: "zh-cn", list: ZH_side() }
const rewrites = handleRewrites(en, zh)
// console.log(rewrites)

export default defineConfig({
    ...shared,
    locales: {
        "en-us": { label: "English", ...en_us },
        root: { label: "简体中文", ...zh_cn },
    },
    // rewrites,
})
