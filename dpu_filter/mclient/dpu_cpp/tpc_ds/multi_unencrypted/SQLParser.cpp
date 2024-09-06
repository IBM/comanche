#include "SQLParser.h"
#include <sstream>
#include <unordered_map>
#include <cctype>
#include <stdexcept>
#include <algorithm>

static const std::unordered_map<std::string, TokenType> tokenMap = {
    {"SELECT", TokenType::SELECT},
    {"FROM", TokenType::FROM},
    {"WHERE", TokenType::WHERE},
    {"AND", TokenType::AND},
    {"OR", TokenType::OR},
    {"BETWEEN", TokenType::BETWEEN},
    {"=", TokenType::OPERATOR},
    {">", TokenType::OPERATOR},
    {">=", TokenType::OPERATOR},
    {"<", TokenType::OPERATOR},
    {"<=", TokenType::OPERATOR},
    {"!=", TokenType::OPERATOR}
};

TokenType SQLParser::getTokenType(const std::string& token) {
    auto it = tokenMap.find(token);
    if (it != tokenMap.end()) {
        return it->second;
    } else if (isColumnName(token)) {
        return TokenType::COLUMN;
    } else if (isLiteral(token)) {
        return TokenType::LITERAL;
    } else {
        return TokenType::UNKNOWN;
    }
}

bool SQLParser::isColumnName(const std::string& token) {
    return std::all_of(token.begin(), token.end(), [](char c) {
        return std::isalnum(c) || c == '_';
    });
}

bool SQLParser::isLiteral(const std::string& token) {
    if (token.empty()) return false;
    if (token.front() == '\'' && token.back() == '\'') return true;
    return std::all_of(token.begin(), token.end(), ::isdigit);
}

void SQLParser::trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

void SQLParser::handleParentheses(const std::string& token, std::vector<Token>& tokens) {
    size_t start = 0;
    while (start < token.length()) {
        size_t openParen = token.find('(', start);
        size_t closeParen = token.find(')', start);

        if (openParen != std::string::npos) {
            if (openParen > start) {
                tokens.push_back({getTokenType(token.substr(start, openParen - start)), token.substr(start, openParen - start)});
            }
            tokens.push_back({TokenType::COLUMN, "("});
            start = openParen + 1;
        } else if (closeParen != std::string::npos) {
            if (closeParen > start) {
                tokens.push_back({getTokenType(token.substr(start, closeParen - start)), token.substr(start, closeParen - start)});
            }
            tokens.push_back({TokenType::COLUMN, ")"});
            start = closeParen + 1;
        } else {
            tokens.push_back({getTokenType(token.substr(start)), token.substr(start)});
            break;
        }
    }
}

std::vector<Token> SQLParser::parse(const std::string& sql) {
    std::vector<Token> tokens;
    std::istringstream stream(sql);
    std::string word;

    while (stream >> word) {
        trim(word);
        handleParentheses(word, tokens);
    }

    return tokens;
}

std::string SQLParser::tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::SELECT: return "SELECT";
        case TokenType::FROM: return "FROM";
        case TokenType::WHERE: return "WHERE";
        case TokenType::COLUMN: return "COLUMN";
        case TokenType::LITERAL: return "LITERAL";
        case TokenType::OPERATOR: return "OPERATOR";
        case TokenType::AND: return "AND";
        case TokenType::OR: return "OR";
        case TokenType::BETWEEN: return "BETWEEN";
        default: return "UNKNOWN";
    }
}
